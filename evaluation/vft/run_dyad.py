from evaluation.vft.interactive import VFT
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser,
)


def load_prompt(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--model_a",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        help="model a name",
    )
    parser.add_argument(
        "--model_b",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        help="model b name",
    )
    parser.add_argument("--model_a_prompt", default="inferred", help="model a prompt")
    parser.add_argument("--model_b_prompt", default="inferred", help="model b prompt")
    parser.add_argument("--turns", default=30, help="number of turns")
    parser.add_argument("--seed", default=42, help="seed")

    args, _ = parser.parse_known_args()
    # print(args)
    # llm_config = None
    # if args.llm_config:
    #     llm_config = get_llm_config_arg(args.llm_config)
    # if llm_config is None:
    #     raise ValueError(f"Could not find LLM config: --llm_config {args.llm_config}")

    # Load the prompts
    system_prompt_a = load_prompt("evaluation/vft/prompts/inferred.txt")
    system_prompt_b = load_prompt("evaluation/vft/prompts/inferred.txt")

    together_api_key = ()
    model_a = args.model_a
    model_a_kargs = {"temperature": 0.5, "max_tokens": 5}
    model_b = args.model_b
    model_b_kargs = {"temperature": 0.5, "max_tokens": 5}
    num_turns = int(args.turns)

    # Initialize the game
    vft = VFT(
        model_a=model_a,
        model_b=model_b,
        system_prompt_a=system_prompt_a,
        system_prompt_b=system_prompt_b,
        num_turns=num_turns,
        together_api=True,
        together_api_key=together_api_key,
        model_a_kargs=model_a_kargs,
        model_b_kargs=model_b_kargs,
        seed=42,
    )
    # Start the game
    vft.start(2)
