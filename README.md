# llm-reasoning-research
Evaluation of LLM reasoning methods using GSM8K

## Overview
大規模言語モデル（LLM）の推論性能向上を目的として、
Chain-of-Thought（CoT）、Tree-of-Thought（ToT）、
Reflexionなどの推論手法を実装・比較しています。

また、探索的推論と反省型自己改善を組み合わせた
推論モデルを構築し、その有効性を検証しています。

## Research Background

LLMは文章生成や質問応答で高い性能を示していますが、
複雑な推論問題では誤った推論を生成する場合があります。

そこで本研究では、複数の推論経路を探索するToTや、
生成した推論を振り返るReflexionなどを組み合わせることで、
推論性能の改善を目指しています。

## Methods

- Chain-of-Thought (CoT)
- Tree-of-Thought (ToT)
- Reflexion
- Proposed Method

## Dataset

GSM8K

## Results

| Method | Accuracy |
|---|---:|
| CoT | 79.6% |
| ToT | 85.9% |
| ToT + Reflexion | 89.6% |
| Proposed Method | 86.9% |

## Discussion

ToTとReflexionを組み合わせることで性能向上を確認した。

一方、提案モデルについては現時点では既存手法を上回る
性能は確認できておらず、原因分析と改善を進めています。

## Future Work

- 推論経路の評価方法の改善
- 学習データの品質改善
- 推論コストと精度の比較
- 提案モデルの追加評価

