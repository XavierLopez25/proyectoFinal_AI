from typing import Any, Dict

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()


def main():
    print("Start...")
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    osu_beatmap_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/osu_beatmap_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    super_mario_64_120_stars_speedrun_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/super_mario_64_120_stars_speedrun_data.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    steam_games_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/steam_games.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    warframe_weapons_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/warframe_weapons.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(name="Python Agent", func=python_agent_executor_wrapper,
             description="""Useful when you need to transform natural language to python and execute the python code,
              returning the results of the code execution DOES NOT ACCEPT CODE AS INPUT"""),
        Tool(name="OSU Beatmap Info Agent", func=osu_beatmap_agent_executor.invoke,
             description="""Useful when you need to answer questions over osu_beatmap_info.csv, specifically questions about OSU beatmaps and questions about 
             Title, Artist, Mapper, Creation Date, Ranked Date, Map Status, Nominator, Genre, Language, Playcount, Likes, Length, BPM, Circle Count, Slider Count,
             Circle Size, HP Drain, Accuracy, Approach Rate, Star Rating, Game Mode, Difficulties, URL of a OSU beatmap, for example: Which is the map with the most
             BPM? Takes an input the entire question and returns the answer after running pandas calculations"""),
        Tool(name="Super Mario 64 - 120 Stars Speedruns Info Agent", func=super_mario_64_120_stars_speedrun_agent_executor.invoke,
             description="""Useful when you need to answer questions over super_mario_64_120_stars_speedrun_data.csv, specifically questions about Super Mario 64 Speedruns to get 120 stars,
              information about the id, place, speedrun link, submitted date, primary time seconds, real time seconds, player id, player name, player country, platform,
              verified of a speedrun, for example: What is the country of the player which has the best time for a speedrun? Takes an input the entire question and returns the answer after running pandas calculations"""),
        Tool(name="Steam Games Info Agent", func=steam_games_agent_executor.invoke,
             description="""This tool is designed to answer queries related to data in the steam_games.csv file. It can handle questions regarding various attributes of Steam games including the game's name,
              developer, publisher, positive and negative ratings, playtime, price, and more. For example, you can ask,
               'Who is the developer of Portal 2?' or 'How many positive ratings does Team Fortress Classic have?' This tool parses natural language questions and maps
                them to the corresponding data fields in the CSV to retrieve accurate information. Takes an input the entire question and returns the answer after running pandas calculations"""),
        Tool(name="Warframe Weapons Info Agent", func=warframe_weapons_agent_executor.invoke,
             description="""Useful when you need to answer questions over warframe_weapons.csv for Warframe weapons, specifically questions about Warframe weapons,
             things like Name, Trigger, AttackName, Impact, Puncture, Slash, Cold, Electricity, Heat, Toxin, Blast, Corrosive, Gas, Magnetic, Radiation, Viral, 
             Void, BaseDamage, BaseDps, TotalDamage, CritChance, CritMultiplier, AvgShotDmg, BurstDps, SustainedDps, LifetimeDmg, StatusChance, ForcedProcs, 
             AvgProcCount, AvgProcPerSec, Multishot, FireRate, ChargeTime, Disposition, Mastery, Magazine, AmmoPickup, AmmoMax, Reload, ShotType, ShotSpeed,
             PunchThrough, Accuracy, Introduced, IntroducedDate, Slot, Class, AmmoType, Range, InternalName, Family, FalloffStart, FalloffEnd, FalloffReduction
             of a weapon, for example: What damage types (Impact, Puncture, Slash, Cold, Electricity, Heat, Toxin, Blast, Corrosive, Gas, Magnetic,
             Radiation, Viral, Void) has the Trumna? Takes an input the entire question and returns the answer after running pandas calculations"""),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )

    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
    )

    # Caso para CSV Agent
    print(
        grand_agent_executor.invoke(
            {
                "input": "What is the name and the country of the person with the fastest Super Mario 64 120 stars speedrun?"
            }
        )
    )
if __name__ == "__main__":
    main()
