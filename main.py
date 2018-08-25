import json
import os
import helper
import sessions
import argparse
from dqn_agent import Agent
from unityagents import UnityEnvironment


def main():
    parser = argparse.ArgumentParser(description="Run Extended Q-Learning with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    args = parser.parse_args()

    # load config files
    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["general"]["env_path"]))
    agent = Agent(config=config)

    if config["train"]["run_training"]:
        scores = sessions.train(agent, env, config)
        helper.plot_scores(scores)
        agent.save()
    else:
        agent.load()
        sessions.test(agent, env)

    env.close()


if __name__ == "__main__":
    main()
