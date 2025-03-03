# LLM Rewording to augment the data

import asyncio
import openai
import pandas as pd
import os
import time

prompt = """
Reword the following text while preserving its meaning and tone:
{text}
---
Here are some examples:
**Text**:
When Prophet Elijah the Tishbite was so frustrated and hopeless that contemplated suicide and told God to kill him , the first thing God did was to solve his immediate need by giving him food . He gave him food again until he ate and his heart was settled and thought of suicide no more .
**Response**:
When Prophet Elijah the Tishbite became so frustrated and despairing that he asked God to end his life, God’s response was unexpectedly practical. Instead of addressing the larger issue right away, He first gave Elijah food, meeting his immediate physical need. God didn’t stop there, providing Elijah with nourishment until he was no longer consumed by his despair and no longer thought of taking his own life.

**Text**:
I am very excited to see the monetary results of having all of these businesses donate toward the cooperative . Since I am also a part of 16xOSU , I have further interest in where the money is going to be used and the difference it will make among the women in Uganda , "" Triplett said ."
**Response**:
”I am absolutely thrilled to see how much money we’ll be able to generate with all these businesses contributing to the cooperative. As a member of 16xOSU, I’m especially keen to see how the funds will be used and the impact they’ll have on the women in Uganda," Triplett remarked.
"""

async def reword_text(client, text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that paraphrases text without changing its meaning."
        },
        {
            "role": "user",
            "content": (prompt.format(text=text))
        },
    ]
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=256
        )

        paraphrased = response.choices[0].message.content.strip()
        return paraphrased

    except Exception as e:
        print("OpenAI API error:", e)
        return text

async def process_texts(client, texts, batch_size=16):
    all_results = []
    for i in range(0, len(texts), batch_size):
        start = time.time()
        print(f"Processing batch {i+1} - {i+batch_size}")

        batch = texts[i: i+batch_size]
        tasks = [asyncio.create_task(reword_text(client, text)) for text in batch]

        res = await asyncio.gather(*tasks)
        all_results.extend(res)
        print(res)

        duration = time.time() - start
        # sleeping = 60 - duration
        print("Took", duration, "seconds")
        time.sleep(60)

    return all_results

async def main():
    print("Enter OpenAI API Key:")
    api_key = input("> ")
    client = openai.AsyncOpenAI(api_key = api_key)

    train_df = pd.read_csv("data/train.csv")

    df_minority = train_df[train_df["label"] == 1].copy()
    df_majority = train_df[train_df["label"] == 0].copy()

    texts = df_minority["text"].tolist()
    reworded_texts = await process_texts(client, texts, batch_size=250)
    df_minority["reworded"] = reworded_texts
    # Drop rows where 'reworded' contains a newline character
    df_minority = df_minority[~df_minority["reworded"].str.contains("\n", na=False)]
    print(df_minority)
    df_minority.to_csv("data/reworded.csv", index=False)

asyncio.run(main())
