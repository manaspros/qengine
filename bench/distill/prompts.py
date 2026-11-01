"""Prompt set for MTP self-distillation (written for this purpose; nothing derived from
bench/prompts/*.txt or bench/prompts/heldout/*: no printing press, TTL/LRU cache, B-tree index or
access-log CSV topics). Deterministic: prompts() returns the same list every time."""
import random

EXPLAIN_TOPICS = [
    "how vaccines train the immune system", "why the sky is blue and sunsets are red", "how a refrigerator moves heat",
    "what inflation is and how central banks fight it", "how GPS receivers compute position", "photosynthesis",
    "how compilers turn source code into machine code", "the causes of the First World War", "how tides work",
    "public-key cryptography", "how noise-cancelling headphones work", "plate tectonics and earthquakes",
    "how a bill becomes law in a parliamentary system", "the water cycle", "how solar panels generate electricity",
    "what a neural network learns during training", "how credit scores are calculated", "the Doppler effect",
    "how the heart pumps blood", "supply and demand", "how garbage collection works in managed languages",
    "black holes", "how a jet engine produces thrust", "the scientific method", "how DNS resolves a domain name",
    "the French Revolution", "how batteries store energy", "entropy in thermodynamics", "how TCP guarantees delivery",
    "climate feedback loops", "how antibiotics work and why resistance develops", "the Roman Republic's government",
    "how an electric motor works", "compound interest", "how memory caches speed up CPUs", "volcanic eruptions",
    "how speech recognition works", "the theory of evolution by natural selection", "how mortgages are amortized",
    "how a transistor switches", "the Silk Road trade network", "how bread rises", "the immune response to allergies",
    "how version control systems track changes", "the Industrial Revolution in textile manufacturing",
    "how airplanes stay in the air", "what a market recession is", "how the eye focuses light", "quantum tunnelling",
]
AUDIENCES = ["a ten-year-old", "a high-school student", "a busy executive", "a university student", "a curious adult"]
CODE_TASKS = [
    ("Python", "parses a Markdown table into a list of dictionaries"), ("Python", "implements Dijkstra's shortest path on a weighted graph"),
    ("Python", "computes rolling statistics over a stream of numbers without storing them all"),
    ("JavaScript", "debounces a function and includes a small usage example"), ("TypeScript", "validates a user registration form object"),
    ("Rust", "reads a text file and counts word frequencies"), ("Go", "implements a concurrent worker pool with a results channel"),
    ("C", "reverses a singly linked list in place"), ("C++", "implements a fixed-size ring buffer class"),
    ("Python", "solves Sudoku with backtracking"), ("SQL", "finds the top three products by revenue per month"),
    ("Bash", "backs up a directory to a timestamped tar archive and deletes archives older than a week"),
    ("Python", "implements a trie with insert, search and prefix listing"), ("Java", "implements a thread-safe bank account"),
    ("Python", "converts Roman numerals to integers and back"), ("JavaScript", "renders a paginated list from a JSON API"),
    ("Python", "implements binary search and a function to find the first occurrence"), ("Rust", "implements a simple tokenizer for arithmetic expressions"),
    ("Python", "simulates a bank queue with random arrivals"), ("Go", "serves a small REST API for a to-do list"),
    ("Python", "merges overlapping intervals"), ("C", "implements matrix multiplication with loop tiling"),
    ("Python", "generates all permutations of a string without duplicates"), ("Kotlin", "models a vending machine state machine"),
    ("Python", "scrapes headings from an HTML string using only the standard library"), ("SQL", "detects duplicate customer records"),
    ("Python", "implements a min-heap from scratch"), ("JavaScript", "implements deep equality of two objects"),
    ("Python", "computes the edit distance between two strings"), ("Haskell", "implements quicksort and explains laziness"),
]
MATH = [
    "A train leaves at {a}:00 travelling {b} km/h; another leaves {c} hours later at {d} km/h on the same track. When does the second catch up?",
    "A rectangle has perimeter {p} cm and its length is {r} times its width. Find its area.",
    "What is the probability of getting at least {k} heads in {n} fair coin flips?",
    "Solve the system: {a}x + {b}y = {c}, {d}x - y = {k}.",
    "A loan of {p} dollars at {r}% annual interest compounded monthly is repaid over {n} years. What is the monthly payment?",
    "Find the sum of all integers from 1 to {p} that are divisible by {a} or {b}.",
    "A recipe for {a} people needs {b} grams of flour. How much flour for {c} people, and how many {d}-gram bags must be bought?",
    "Prove that the sum of the first n odd numbers is n squared, then check it for n = {k}.",
]
WRITING = [
    "Write a short story about {x}.", "Write a polite email declining {x}.", "Write a poem about {x}.",
    "Write a product description for {x}.", "Write a persuasive paragraph arguing for {x}.",
    "Write a dialogue between two friends discussing {x}.", "Summarize the pros and cons of {x} in a short essay.",
]
WRITING_X = [
    "a lighthouse keeper who finds a message in a bottle", "an invitation to speak at a conference", "autumn in a small town",
    "a solar-powered backpack", "a four-day work week", "learning a musical instrument as an adult", "remote work",
    "a robot learning to cook", "a job offer that requires relocation", "the first snowfall", "an electric bicycle",
    "public libraries", "moving to a new city", "a cat that hates Mondays", "city gardens on rooftops",
]
CHAT = [
    "How can I get better at waking up early?", "What should I pack for a week of hiking in the mountains?",
    "My team keeps missing deadlines. How do I talk to them about it?", "How do I start learning statistics on my own?",
    "What are good habits for keeping a tidy home with little time?", "How do I prepare for a job interview for a data analyst role?",
    "I want to run a 10 km race in three months. Give me a plan.", "How should I compare two job offers?",
    "What is a balanced weekly meal plan on a budget?", "How do I give constructive feedback to a colleague?",
    "How can I reduce my screen time?", "What should I consider before adopting a dog?",
    "How do I plan a three-day trip to a city I have never visited?", "How do I negotiate a higher salary?",
    "What are some ways to make a small apartment feel bigger?", "How do I teach a child to ride a bicycle?",
]
REASON = [
    "Three friends split a bill unevenly: {a}, {b} and {c} dollars. They want to share it equally. Who pays whom and how much?",
    "If all bloops are razzies and some razzies are lazzies, can we conclude that some bloops are lazzies? Explain.",
    "A farmer must cross a river with a wolf, a goat and a cabbage. Explain a safe plan step by step.",
    "You have {a} litres and {b} litres jugs and need exactly {c} litres. Is it possible? Show the steps or prove it is not.",
    "Schedule {a} meetings of {b} minutes each between 9:00 and 17:00 with {c}-minute breaks. Is there enough time?",
]
DATA = [
    "Design a database schema for {x} and explain the relationships.", "Explain how you would test {x}.",
    "Compare two architectures for {x} and recommend one.", "Write a project plan for building {x}.",
]
DATA_X = ["a library lending system", "a ride-sharing app", "an online course platform", "a hospital appointment system",
          "a recipe sharing website", "a warehouse inventory tracker", "a chat application", "a fitness tracking app"]


def prompts(seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    out = []
    for t in EXPLAIN_TOPICS:
        for aud in rng.sample(AUDIENCES, 2):
            out.append(f"Explain {t} to {aud}. Use an example and keep it accurate.")
    for lang, task in CODE_TASKS:
        out.append(f"Write a {lang} program that {task}. Include comments and explain the approach.")
        out.append(f"In {lang}, write code that {task}, then describe its time complexity and edge cases.")
    for tpl in MATH:
        for _ in range(6):
            v = dict(a=rng.randint(2, 9), b=rng.randint(20, 120), c=rng.randint(2, 9), d=rng.randint(40, 160),
                     p=rng.randint(30, 900), r=rng.randint(2, 6), k=rng.randint(2, 7), n=rng.randint(5, 30))
            out.append(tpl.format(**v) + " Show your reasoning.")
    for tpl in WRITING:
        for x in rng.sample(WRITING_X, 5):
            out.append(tpl.format(x=x))
    out += CHAT
    for tpl in REASON:
        for _ in range(4):
            out.append(tpl.format(a=rng.randint(3, 13), b=rng.randint(15, 60), c=rng.randint(4, 30)))
    for tpl in DATA:
        for x in DATA_X:
            out.append(tpl.format(x=x))
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    p = prompts()
    print(len(p))
    for x in p[:5]:
        print("-", x)
