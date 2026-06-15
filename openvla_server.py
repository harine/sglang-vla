import numpy as np
from transformers import AutoConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import json_numpy as json
import uvicorn
import os
import argparse

import sglang as sgl


ACTION_DIM = 7
MAX_ACTION_GENERATION_ATTEMPTS = 3


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen(
        "action",
        max_tokens=ACTION_DIM,
        min_tokens=ACTION_DIM,
        ignore_eos=True,
        return_logprob=True,
    )


class TokenToAction:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def convert(self, output_ids):
        predicted_action_token_ids = np.array(output_ids)
        if predicted_action_token_ids.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Expected {ACTION_DIM} action tokens, got {predicted_action_token_ids.shape[-1]}: "
                f"{predicted_action_token_ids.tolist()}"
            )
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions


class BatchRequest(BaseModel):
    instructions: List[str]
    image_path: str
    temperature: float = 1.0


class BatchResponse(BaseModel):
    output_ids: List[List[int]]
    actions: List[List[float]]


# Initialize the converter
converter = TokenToAction()

# Initialize FastAPI app
app = FastAPI(title="VLA Action Server", description="Server for generating robot actions from vision-language instructions")

# Global runtime variable
runtime = None


@app.on_event("startup")
async def startup_event():
    """Initialize the sglang runtime on startup"""
    global runtime
    # Get seed from environment variable (set in __main__)
    seed = int(os.environ.get("VLA_RANDOM_SEED", "42"))
    rt_kwargs = dict(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=seed,
        trust_remote_code=True,
    )
    # Optionally cap the static memory fraction so the server can share a GPU
    # with another process (e.g. the in-process verifier + SAPIEN eval on a
    # single H200). Unset -> sglang default.
    _mf = os.environ.get("VLA_MEM_FRACTION", "").strip()
    if _mf:
        rt_kwargs["mem_fraction_static"] = float(_mf)
        print(f"[openvla_server] mem_fraction_static={_mf}")
    runtime = sgl.Runtime(**rt_kwargs)
    sgl.set_default_backend(runtime)
    print(f"SGLang runtime initialized successfully with seed: {seed}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the sglang runtime"""
    global runtime
    if runtime:
        runtime.shutdown()
        print("SGLang runtime shutdown successfully")


@app.post("/batch", response_model=BatchResponse)
async def batch_actions(request: BatchRequest):
    """
    Process batch of instructions and return action tokens and continuous actions.
    
    Args:
        request: BatchRequest containing instructions, image_path, and temperature
        
    Returns:
        BatchResponse with output_ids and actions for each instruction
    """
    try:
        # Validate image path exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail=f"Image file not found: {request.image_path}")
        
        all_output_ids = [None] * len(request.instructions)
        all_actions = [None] * len(request.instructions)
        pending = list(enumerate(request.instructions))

        for attempt in range(1, MAX_ACTION_GENERATION_ATTEMPTS + 1):
            if not pending:
                break

            arguments = []
            for _, instruction in pending:
                question = f"In: What action should the robot take to {instruction}?\nOut:"
                arguments.append({
                    "image_path": request.image_path,
                    "question": question,
                })

            states = image_qa.run_batch(
                arguments,
                temperature=request.temperature,
            )

            retry_pending = []
            for (request_idx, instruction), state in zip(pending, states):
                output_logprobs = state.get_meta_info("action")["output_token_logprobs"]
                output_ids = [logprob[1] for logprob in output_logprobs]

                try:
                    actions = converter.convert(output_ids)
                except ValueError as e:
                    print(
                        f"Invalid action generation on attempt {attempt}/"
                        f"{MAX_ACTION_GENERATION_ATTEMPTS}: {e}"
                    )
                    retry_pending.append((request_idx, instruction))
                    continue

                all_output_ids[request_idx] = output_ids
                all_actions[request_idx] = actions.tolist()

            pending = retry_pending

        if pending:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to generate {ACTION_DIM} action tokens for "
                    f"{len(pending)} / {len(request.instructions)} requests after "
                    f"{MAX_ACTION_GENERATION_ATTEMPTS} attempts"
                ),
            )
        
        return BatchResponse(
            output_ids=all_output_ids,
            actions=all_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "runtime_initialized": runtime is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Action Server")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set environment variable so it's accessible when uvicorn imports the module
    os.environ["VLA_RANDOM_SEED"] = str(args.seed)
    
    uvicorn.run(
        "openvla_server:app",
        host="0.0.0.0",
        port=3200,
        reload=False,
        log_level="info"
    )
