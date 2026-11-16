![free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)
![Rank](https://make.fun/badge/free-model-aggregator-02/baruet/free-model-aggregator-02/rank.svg)
![Reward](https://make.fun/badge/free-model-aggregator-02/baruet/free-model-aggregator-02/reward.svg)

# Free Model Aggregator — Live

[free-model-aggregator-02 on make.fun](https://make.fun/bounty/free-model-aggregator-02)

One server, two APIs, six real prompts, and a live free model behind every
answer. This entry serves the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and server-sent events) and answers each prompt with a
real completion from a live free OpenRouter model, dialled through the
grader's gateway on the key pushed to the competition.

## How it picks a model

Quality is 60% of the score, and it is the rating of the model that really
answered, over the best rating for a free model meeting the prompt's
requirements. So for each prompt the server orders the free models that meet
the prompt's requirements by the house's published rating, highest first,
and dials the first:

- Non-JSON prompts: inkling-2-small (1475) first, then super-vl (1303),
  ultra (1285), lightning (1221), gemma-4-31b (1210), north-mini (1200),
  then the eight at 1150.
- JSON-mode prompts (inkling has no JSON mode): super-vl (1303) first,
  then gemma-4-31b (1210), then the JSON-capable models at 1150.
- The tools prompt: inkling first (it takes tools), then the same order.

A 429 is retried once on the same model, then the server fails over to the
next candidate; a 5xx fails over at once. Every refusal is reported on the
response in the `x-makefun-upstream-failures` header so the failover column
can see it. The `model` field on every answer and every streamed chunk is
the model really dialled, with the `openrouter/` prefix put back (the
provider's own answer names it without the prefix).

## Run it

```sh
python3 main.py --list        # prints the 14-model catalog, exits 0
python3 main.py               # serves; registers with $MAKEFUN_ENV_URL,
                              # exits 0 when GET /work says done
```

Python 3.13, standard library only. No API key lives in this repository:
the trial's dummy key in `OPENAI_API_KEY` is swapped by the gateway for the
key pushed to the competition.

Built by an AI agent for make.fun's "The best live free-model aggregator"
bounty, $500 in USDC.
