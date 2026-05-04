import json
import os
import random

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}},
            "required": ["city"]
        }
    },
    {
        "name": "book_flight",
        "description": "Book a flight from one city to another.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "calculate_mortgage",
        "description": "Calculate mortgage payment.",
        "parameters": {
            "type": "object",
            "properties": {"principal": {"type": "number"}, "rate": {"type": "number"}, "years": {"type": "integer"}},
            "required": ["principal", "rate", "years"]
        }
    }
]

TEMPLATES = [
    ("What's the weather like in {city}?", "get_weather", {"city": "{city}"}),
    ("I want to fly from {city1} to {city2} on {date}.", "book_flight", {"origin": "{city1}", "destination": "{city2}", "date": "{date}"}),
    ("Calculate my mortgage for ${amount} at {rate}% over {years} years.", "calculate_mortgage", {"principal": "{amount}", "rate": "{rate}", "years": "{years}"})
]

CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto"]
DATES = ["2023-12-01", "2024-01-15", "tomorrow", "next week"]
AMOUNTS = [100000, 250000, 500000, 1000000]
RATES = [3.5, 4.0, 5.25, 6.0]
YEARS = [15, 20, 30]

def generate_example():
    template, func_name, args_template = random.choice(TEMPLATES)
    city = random.choice(CITIES)
    city1 = random.choice(CITIES)
    city2 = random.choice(CITIES)
    date = random.choice(DATES)
    amount = random.choice(AMOUNTS)
    rate = random.choice(RATES)
    years = random.choice(YEARS)
    
    user_query = template.format(city=city, city1=city1, city2=city2, date=date, amount=amount, rate=rate, years=years)
    
    args = {}
    for k, v in args_template.items():
        val = v.format(city=city, city1=city1, city2=city2, date=date, amount=amount, rate=rate, years=years)
        if v == "{amount}": val = amount
        elif v == "{rate}": val = rate
        elif v == "{years}": val = years
        args[k] = val
        
    system_prompt = f"You are a helpful assistant with access to the following functions. Use them if required -\n{json.dumps(TOOLS, indent=4)}"
    assistant_content = json.dumps({"name": func_name, "arguments": args})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def main():
    print("Generating synthetic data...")
    results = [generate_example() for _ in range(1300)]
    
    train = results[:1000]
    valid = results[1000:1100]
    test = results[1100:1300]
    
    os.makedirs("data", exist_ok=True)
    
    def save_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                
    save_jsonl(train, "data/train.jsonl")
    save_jsonl(valid, "data/valid.jsonl")
    save_jsonl(test, "data/test.jsonl")
    
    print("Saved data/train.jsonl (1000), data/valid.jsonl (100), data/test.jsonl (200)")

if __name__ == "__main__":
    main()
