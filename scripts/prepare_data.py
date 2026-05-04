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
    },
    {
        "name": "search_restaurants",
        "description": "Search for restaurants in a city by cuisine.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "cuisine": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["city", "cuisine"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "exchange": {"type": "string"}},
            "required": ["ticker"]
        }
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "source_lang": {"type": "string"}, "target_lang": {"type": "string"}},
            "required": ["text", "target_lang"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}, "duration_minutes": {"type": "integer"}},
            "required": ["title", "date", "time"]
        }
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist on a music streaming service.",
        "parameters": {
            "type": "object",
            "properties": {"song": {"type": "string"}, "artist": {"type": "string"}, "service": {"type": "string"}},
            "required": ["song"]
        }
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a given task at a specific time.",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string"}, "datetime": {"type": "string"}},
            "required": ["task", "datetime"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for a given query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
            "required": ["query"]
        }
    }
]

CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Mumbai", "Dubai", "Singapore"]
DATES = ["2024-03-01", "2024-04-15", "tomorrow", "next Monday", "2024-06-20"]
AMOUNTS = [100000, 250000, 500000, 1000000]
RATES = [3.5, 4.0, 5.25, 6.0]
YEARS = [15, 20, 30]
CUISINES = ["Italian", "Japanese", "Mexican", "Indian", "French", "Thai", "Chinese"]
TICKERS = ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN"]
LANGS = ["French", "German", "Spanish", "Japanese", "Arabic", "Hindi"]
SONGS = ["Blinding Lights", "Shape of You", "Bohemian Rhapsody", "Hotel California"]
ARTISTS = ["The Weeknd", "Ed Sheeran", "Queen", "Eagles"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "INR", "AED"]
TIMES = ["09:00", "14:30", "18:00", "10:00"]
PHRASES = ["Good morning", "How are you?", "Thank you very much", "Where is the nearest hospital?"]

TEMPLATES = [
    lambda: (
        f"What's the weather like in {random.choice(CITIES)}?",
        "get_weather",
        {"city": None}  # filled inline
    ),
    lambda: (
        None, "book_flight", None  # handled inline
    ),
    lambda: (
        f"Calculate my mortgage for ${random.choice(AMOUNTS)} at {random.choice(RATES)}% over {random.choice(YEARS)} years.",
        "calculate_mortgage",
        None
    ),
    lambda: (
        f"Find {random.choice(CUISINES)} restaurants in {random.choice(CITIES)}.",
        "search_restaurants",
        None
    ),
    lambda: (
        f"What is the current price of {random.choice(TICKERS)}?",
        "get_stock_price",
        None
    ),
    lambda: (
        f"Translate '{random.choice(PHRASES)}' to {random.choice(LANGS)}.",
        "translate_text",
        None
    ),
    lambda: (
        f"Convert {random.randint(100, 10000)} {random.choice(CURRENCIES)} to {random.choice(CURRENCIES)}.",
        "convert_currency",
        None
    ),
    lambda: (
        f"Play {random.choice(SONGS)} by {random.choice(ARTISTS)}.",
        "play_music",
        None
    ),
    lambda: (
        f"Remind me to check emails at {random.choice(TIMES)} tomorrow.",
        "set_reminder",
        None
    ),
    lambda: (
        f"Search the web for 'best {random.choice(CUISINES)} recipes'.",
        "search_web",
        None
    ),
]

def generate_example():
    tool_index = random.randint(0, 11)
    tool = TOOLS[tool_index]
    func_name = tool["name"]

    if func_name == "get_weather":
        city = random.choice(CITIES)
        user_query = f"What's the weather like in {city}?"
        args = {"city": city}

    elif func_name == "book_flight":
        cities = random.sample(CITIES, 2)  # Guarantees city1 != city2
        date = random.choice(DATES)
        user_query = f"I want to fly from {cities[0]} to {cities[1]} on {date}."
        args = {"origin": cities[0], "destination": cities[1], "date": date}

    elif func_name == "calculate_mortgage":
        principal = random.choice(AMOUNTS)
        rate = random.choice(RATES)
        years = random.choice(YEARS)
        user_query = f"Calculate my mortgage for ${principal} at {rate}% over {years} years."
        args = {"principal": principal, "rate": rate, "years": years}

    elif func_name == "search_restaurants":
        city = random.choice(CITIES)
        cuisine = random.choice(CUISINES)
        user_query = f"Find {cuisine} restaurants in {city}."
        args = {"city": city, "cuisine": cuisine}

    elif func_name == "get_stock_price":
        ticker = random.choice(TICKERS)
        user_query = f"What is the current price of {ticker}?"
        args = {"ticker": ticker}

    elif func_name == "translate_text":
        phrase = random.choice(PHRASES)
        lang = random.choice(LANGS)
        user_query = f"Translate '{phrase}' to {lang}."
        args = {"text": phrase, "target_lang": lang}

    elif func_name == "send_email":
        user_query = "Send an email to boss@company.com with subject 'Meeting Tomorrow' and body 'Please confirm attendance.'"
        args = {"to": "boss@company.com", "subject": "Meeting Tomorrow", "body": "Please confirm attendance."}

    elif func_name == "create_calendar_event":
        title = random.choice(["Team Standup", "Doctor Appointment", "Lunch with Client", "Weekly Review"])
        date = random.choice(DATES)
        time = random.choice(TIMES)
        user_query = f"Create a calendar event '{title}' on {date} at {time}."
        args = {"title": title, "date": date, "time": time}

    elif func_name == "convert_currency":
        amount = random.randint(100, 10000)
        currencies = random.sample(CURRENCIES, 2)
        user_query = f"Convert {amount} {currencies[0]} to {currencies[1]}."
        args = {"amount": amount, "from_currency": currencies[0], "to_currency": currencies[1]}

    elif func_name == "play_music":
        song = random.choice(SONGS)
        artist = random.choice(ARTISTS)
        user_query = f"Play {song} by {artist}."
        args = {"song": song, "artist": artist}

    elif func_name == "set_reminder":
        tasks = ["check emails", "call the dentist", "pay rent", "submit report"]
        task = random.choice(tasks)
        time = random.choice(TIMES)
        user_query = f"Remind me to {task} at {time} tomorrow."
        args = {"task": task, "datetime": f"tomorrow {time}"}

    else:  # search_web
        query = random.choice([f"best {random.choice(CUISINES)} recipes", f"how to learn {random.choice(LANGS)}", "top 10 programming languages"])
        user_query = f"Search the web for '{query}'."
        args = {"query": query}

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
    random.seed(42)
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
