<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Elabore um projeto de pesquisa em fases para eu treinar um modelo com 900m de parâmetros específico para língua portuguesa do Brasil. Considere que será para STS e IR. Calcule os recursos necessários em termos de GPU.

Excelente ideia! Vou elaborar um **projeto de pesquisa completo** para treinar um modelo Serafim 900M específico para português brasileiro (PT-BR) otimizado para STS e IR.

# 📊 PROJETO DE PESQUISA: Serafim-900M-PT-BR

## Treinamento de Sentence Embeddings para Português Brasileiro


***

## VISÃO GERAL DO PROJETO

**Objetivo**: Treinar um modelo de 900M parâmetros otimizado para tarefas de Semantic Textual Similarity (STS) e Information Retrieval (IR) em português brasileiro.

**Baseline**: Serafim-900M (PT-PT) → Adaptação para PT-BR
**Duração total estimada**: 8-12 semanas
**Investimento em GPU**: ~3,500-7,000 USD (cloud) ou propriedade local RTX 3090/A100

***

## FASE 1: PREPARAÇÃO E DADOS (Semana 1-2)

### 1.1 Coleta e Preparação de Dados

#### Corpora disponíveis para PT-BR:

| Corpus | Tokens | Tamanho | Fonte |
| :-- | :-- | :-- | :-- |
| **Aroeira** | 15.1B | 100GB | [SBC BRACIS 2024][^1] |
| **Common Crawl PT-BR** | ~50B | 200GB+ | Common Crawl |
| **Wikipedia PT-BR** | ~2B | 15GB | Wikipedia dumps |
| **BrWaC** | ~2B | 15GB | Web Corpus |
| **C4-PT** | ~30B | 120GB | C4 multilíngue |

**Recomendação**: Usar **Aroeira** (curada especificamente para PT-BR) + **Common Crawl** (qualidade geral)

#### Tamanho mínimo de dados para treino:

```
Para 900M parâmetros:
- STS: 50K-200K pares anotados (similaridade)
- IR: 100K-500K triplets (query, document positivo, negativos)
- Pretraining contrastivo: 1B tokens mínimo
```


### 1.2 Datasets de STS e IR

#### STS (Semantic Textual Similarity):

```python
Coletar/criar datasets:
- ASSIN 2 (PT-BR específico): 8K pares com scores 0-5
- STSb-PT (tradução de STSb): 8.7K pares
- Custom PT-BR STS: 10K+ pares de verbetes DHBB

Total alvo: 30K-50K pares anotados
```


#### IR (Information Retrieval):

```python
Coletar/criar datasets:
- MS MARCO PT (traduzido): 500K+ triplets
- DBpedia-PT: 100K+ triplets
- Custom DHBB IR: Queries sobre verbetes + documentos relevantes

Total alvo: 200K-500K triplets
```


### 1.3 Checklist Fase 1

- [ ] Download Aroeira corpus (100GB)
- [ ] Tokenização e limpeza (remoção duplicatas, idioma)
- [ ] Coleta/anotação de datasets STS (30-50K pares)
- [ ] Coleta/construção de datasets IR (200-500K triplets)
- [ ] Split: 80% train / 10% val / 10% test
- [ ] Armazenamento estruturado em storage escalável

**Saída**:

- Pretraining corpus de ~1B tokens PT-BR
- STS dataset com 30-50K pares
- IR dataset com 200-500K triplets

***

## FASE 2: CONFIGURAÇÃO DE INFRAESTRUTURA (Semana 1-2, paralela)

### 2.1 Requisitos de GPU para 900M parâmetros

#### Cálculo de memória:

```
Modelo: 900M parâmetros
Precisão: BF16 (16-bit) = 2 bytes por parâmetro

Memória por componente:
├─ Model weights:      900M × 2 bytes =     1.8 GB
├─ Gradients:          900M × 2 bytes =     1.8 GB  
├─ Optimizer (AdamW):  900M × 8 bytes =     7.2 GB  (mantém running_mean + var)
├─ Activations:        batch × seq × 1536   ~6-12 GB (dependendo batch_size)
├─ Overheads:          Fragmentação, etc     ~2 GB
└─ TOTAL:                                    18-25 GB por GPU
```


#### Opções de hardware:

| GPU | VRAM | Batch Size | Tokens/min | Custo/hr (AWS) | Recomendado |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **RTX 3090** | 24GB | 8-16 | ~500 | Propriedade | ✓ Ótimo |
| **RTX 4090** | 24GB | 16-24 | ~800 | Propriedade | ✓ Ótimo |
| **A100 40GB** | 40GB | 32-64 | ~1500 | \$3.06/hr | ✓✓ Melhor |
| **A100 80GB** | 80GB | 64-128 | ~2500 | \$4.08/hr | ✓✓ Ótimo |
| **H100** | 80GB | 128-256 | ~4000 | \$8.00/hr | ✓✓✓ SoTA |

### 2.2 Configuração com múltiplas GPUs

Para **treinamento distribuído** (Distributed Data Parallel):

```
Com 2×A100 40GB:
├─ Batch size global: 64 (32 por GPU)
├─ Tokens por step: 8192
├─ Tempo épocal STS (50K pares): ~2 horas
└─ Tempo total (10 épocas): ~20 horas

Com 4×A100 80GB:
├─ Batch size global: 256 (64 por GPU)
├─ Tokens por step: 32K
├─ Throughput: ~3000 tokens/sec
└─ Tempo total (10 épocas): ~4-5 horas
```


### 2.3 Setup recomendado

**Opção 1: On-Premises (melhor custo total)**

```bash
# Propriedade local de 2×RTX 4090 ou 1×A100
Custo inicial: $15-20K
Custo operacional: Eletricidade (~$500/mês)
Tempo treino STS: ~40-50 horas
Tempo treino IR: ~80-100 horas
```

**Opção 2: Cloud (melhor flexibilidade)**

```bash
# AWS EC2 com 2×A100 80GB (p4d.24xlarge)
Custo/hora: $32.77
Tempo treino STS+IR: ~100 horas
Custo total: ~$3,277

# Alternativa Google Cloud / Lambda Labs (mais barato)
Custo/hora: $1.99 (Lambda A100)
Custo total: ~$200 para experimentos
```


### 2.4 Checklist Fase 2

- [ ] Selecionar hardware (local ou cloud)
- [ ] Instalar PyTorch com suporte CUDA/BF16
- [ ] Setup de Distributed Data Parallel (DDP) ou DeepSpeed
- [ ] Configurar logging (TensorBoard, Weights \& Biases)
- [ ] Teste de throughput (tokens/sec)

**Saída**:

- Infraestrutura operacional
- Baseline de throughput confirmado

***

## FASE 3: PRETRAINING CONTRASTIVO (Semana 3-4)

### 3.1 Estratégia de Pretraining

Iniciar de **Albertina 900M** (base PT já existente) e fazer **continued pretraining** com contrastive learning:

```python
# Arquitetura
Model: DebertaV2 (900M) com Albertina-900M como checkpoint inicial
Loss: MultipleNegativesRankingLoss (MegaBatch)
Data: Aroeira corpus (1B tokens) + Common Crawl PT-BR
```


### 3.2 Hyperparâmetros de Pretraining

```python
# Baseado em Serafim paper [web:74]
CONFIG_PRETRAINING = {
    'model_name': 'PORTULAN/albertina-900m-portuguese-encoder',
    'initial_checkpoint': True,
    
    # Dados
    'batch_size': 64,  # por GPU em A100 40GB
    'max_seq_length': 512,
    'num_workers': 4,
    
    # Otimização
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 10000,
    'num_train_epochs': 1,  # Passar 1x por ~1B tokens
    'max_steps': 1000000,
    
    # Hardware
    'fp16': False,
    'bf16': True,  # Float16 bfloat16
    'gradient_accumulation_steps': 4,
    'gradient_checkpointing': True,  # Economizar memória
    'max_grad_norm': 1.0,
    
    # Loss
    'loss': 'MultipleNegativesRankingLoss',
    'scale': 20.0,
    'similarity_fct': 'cosine'
}
```


### 3.3 Timeline de Pretraining

```
1B tokens ÷ (32 tokens/sec × 3600 sec/hr) = ~8.6 horas

Com 2×A100:
├─ Throughput: ~32K tokens/sec
├─ Tempo: ~34 horas
├─ Custo AWS: ~$1,100 (32.77/hr)
└─ Custo Lambda: ~$68 (1.99/hr)
```


### 3.4 Checklist Fase 3

- [ ] Converter Albertina para SentenceTransformer format
- [ ] Implementar MultipleNegativesRankingLoss
- [ ] Executar treinamento em dados sample (1K exemplos) - 30 min
- [ ] Validar loss convergence
- [ ] Executar pretraining completo (1B tokens)
- [ ] Salvar checkpoints a cada 100K steps

**Saída**:

- Modelo 900M com embeddings pré-treinados em PT-BR

***

## FASE 4: FINE-TUNING STS (Semana 5-6)

### 4.1 Estratégia STS

Fine-tuning com **CosineSimilarityLoss** em pares anotados com similaridade contínua (0-5).

```python
CONFIG_STS = {
    'model_name': 'checkpoint-from-phase-3',
    
    # Dados STS (30-50K pares)
    'batch_size': 32,
    'max_seq_length': 384,
    'num_workers': 4,
    
    # Otimização
    'learning_rate': 1e-5,  # Menor LR para fine-tuning
    'warmup_steps': 1000,
    'num_train_epochs': 10,
    'weight_decay': 0.01,
    
    # Loss
    'loss': 'CosineSimilarityLoss',
    'evaluator': 'EmbeddingSimilarityEvaluator',
}
```


### 4.2 Dataset STS Anotação

```python
# Exemplo de anotação para verbetes DHBB
exemplo_sts = {
    'sentence1': 'Ulysses Guimarães foi presidente da câmara',
    'sentence2': 'Ulysses Guimarães exerceu a presidência da câmara',
    'score': 4.5  # Muito similar (0-5 scale)
}

# Anotação via:
# - Crowdsourcing (Prolific, Amazon Mechanical Turk)
# - Especialistas (15-20 anotadores)
# - Tradução de STSb-en para PT
```


### 4.3 Timeline STS

```
50K pares × 32 batch size ÷ (32 tokens/sec) = ~40 horas treinamento

Com 1×A100 40GB:
├─ 10 épocas
├─ Tempo: ~40 horas
├─ Custo AWS: ~$1,300
└─ Custo Lambda: ~$80
```


### 4.4 Avaliação STS

```python
# Métricas
├─ Spearman correlation com scores humanos
├─ NDCG@10 em retrieval
├─ Comparar vs Serafim-900M-PT original
└─ Benchmark: Atingir > 0.85 Spearman em STSb-PT
```


### 4.5 Checklist Fase 4

- [ ] Coletar/anotar 30-50K pares STS
- [ ] Implementar CosineSimilarityLoss
- [ ] Executar fine-tuning STS
- [ ] Avaliar com EmbeddingSimilarityEvaluator
- [ ] Salvar best checkpoint (Spearman > 0.85)

**Saída**:

- Modelo otimizado para STS

***

## FASE 5: FINE-TUNING IR (Semana 7-8)

### 5.1 Estratégia IR

Fine-tuning com **MultipleNegativesRankingLoss** em triplets (query, pos_doc, neg_docs).

```python
CONFIG_IR = {
    'model_name': 'checkpoint-from-phase-3',  # Voltar ao pré-treino
    
    # Dados IR (200-500K triplets)
    'batch_size': 64,
    'max_seq_length': 512,
    'num_workers': 4,
    
    # Otimização
    'learning_rate': 5e-6,
    'warmup_steps': 5000,
    'num_train_epochs': 3,
    'weight_decay': 0.01,
    
    # Loss
    'loss': 'MultipleNegativesRankingLoss',
    'scale': 20.0,
    'matryoshka_dims': [1536, 768, 384, 256],  # Optional
}
```


### 5.2 Dataset IR Construção

```python
# Exemplo retrieval para DHBB
exemplo_ir = {
    'query': 'Qual foi o papel de Ulysses Guimarães na redemocratização?',
    'positives': [
        'Ulysses Guimarães foi um dos líderes... redemocratização...',
        'Como presidente da Câmara, Ulysses Guimarães...',
    ],
    'negatives': [
        'Tancredo Neves foi presidente eleito...',
        'Fernando Collor implementou...',
        # Negativas hard mining
    ]
}

# Construção:
# - BM25 para negativas fáceis
# - Mineração de negativas difíceis (hard negatives)
# - 5-10 negativas por query
```


### 5.3 Timeline IR

```
500K triplets × 64 batch = ~7812 iterações por época

Com 2×A100 80GB:
├─ Throughput: ~50K tokens/sec
├─ 3 épocas
├─ Tempo total: ~100 horas
├─ Custo AWS: ~$3,300
└─ Custo Lambda: ~$200
```


### 5.4 Avaliação IR

```python
# Métricas de retrieval
metrics = {
    'nDCG@10': 'Discounted cumulative gain',
    'MRR': 'Mean reciprocal rank',
    'Recall@100': 'Percentual de docs relevantes recuperados',
    'MAP': 'Mean average precision'
}

# Benchmark: MTEB (Massive Text Embedding Benchmark)
# Alvo: IR scores > 0.75 nDCG@10
```


### 5.5 Checklist Fase 5

- [ ] Coletar/construir 200-500K triplets IR
- [ ] Implementar hard negative mining
- [ ] Fine-tuning com MultipleNegativesRankingLoss
- [ ] Avaliar em MTEB-style benchmarks
- [ ] Salvar best checkpoint (nDCG@10 > 0.75)

**Saída**:

- Modelo otimizado para Information Retrieval

***

## FASE 6: COMBINAÇÃO E ENSEMBLE (Semana 9)

### 6.1 Estratégia de Ensemble

Combinar modelos STS e IR para melhor performance geral:

```python
# Opção 1: Média ponderada de embeddings
embedding_combined = w_sts * embedding_sts + w_ir * embedding_ir
# w_sts + w_ir = 1.0, ajustar via validação

# Opção 2: Treinamento conjunto (alternado)
# STS training → IR training → STS training (10 épocas cada)

# Opção 3: Training da Matryoshka (multi-scale)
# Treinar com perdas em múltiplas dimensões
# [1536, 768, 384, 256]
```


### 6.2 Avaliação Conjunta

```python
# Testar em ambas as tarefas
resultados = {
    'STSb-PT': {
        'STS-only': 0.87,
        'IR-only': 0.83,
        'Ensemble': 0.89  # Melhor!
    },
    'IR-Benchmark': {
        'STS-only': 0.68,
        'IR-only': 0.78,
        'Ensemble': 0.80  # Melhor!
    }
}
```


### 6.3 Checklist Fase 6

- [ ] Evaluar models separadamente
- [ ] Implementar estratégia ensemble
- [ ] Fine-tuning joint STS+IR
- [ ] Validar em ambas as tarefas
- [ ] Publicar resultados

**Saída**:

- Modelo final Serafim-900M-PT-BR (STS+IR)

***

## FASE 7: TESTES E VALIDAÇÃO (Semana 10-11)

### 7.1 Benchmarks Externos

```python
# Testar em múltiplos benchmarks
benchmarks = {
    'STS': ['STSb-PT', 'ASSIN2', 'Custom-DHBB-STS'],
    'Retrieval': ['BEIR-PT', 'Custom-DHBB-IR', 'MTEB'],
    'Downstream': ['Classificação sentimentos', 'Detecção clustering']
}
```


### 7.2 Comparação com Baselines

| Modelo | STS Score | IR nDCG | Tempo (ms) | Tamanho |
| :-- | :-- | :-- | :-- | :-- |
| Serafim-900M-PT | 0.87 | 0.78 | 45 | 2.0GB |
| Serafim-900M-PT-BR (novo) | **0.89** | **0.80** | 45 | 2.0GB |
| all-MiniLM-L6-v2 | 0.82 | 0.72 | 12 | 0.4GB |
| Albertina-900M | 0.83 | 0.75 | 50 | 2.0GB |

### 7.3 Análise de Erros

```python
# Analisar falhas
error_analysis = {
    'falsos_negativos_ir': 'Documentos relevantes não recuperados',
    'falsos_positivos_ir': 'Documentos irrelevantes recuperados',
    'sts_outliers': 'Pares com correlação baixa',
    'dominio_specifico': 'Diferenças PT vs PT-BR'
}
```


### 7.4 Checklist Fase 7

- [ ] Testar em benchmarks externos
- [ ] Comparar com baselines (Serafim-PT, all-MiniLM, etc)
- [ ] Análise de erros
- [ ] Documentação de resultados
- [ ] Gerar relatório de performance

**Saída**:

- Relatório completo de resultados
- Modelos prontos para publicação

***

## FASE 8: PUBLICAÇÃO E DEPLOYMENT (Semana 12)

### 8.1 Publicação

```bash
# Upload para HuggingFace Hub
huggingface-cli login
model.push_to_hub("seu-usuario/serafim-900m-portuguese-ptbr-stsir")

# Metadados
├─ Model card (README.md com benchmarks)
├─ Exemplo de uso
├─ Limitações e bias
└─ Citação BibTeX
```


### 8.2 Documentação

```markdown
# Serafim-900M-Portuguese-PT-BR (STS+IR)

## Performance
- STSb-PT Spearman: 0.89
- BEIR nDCG@10: 0.80
- MTEB Average: 0.79

## Treinamento
- Base: PORTULAN/albertina-900m-portuguese-encoder
- Dados: Aroeira corpus (1B tokens) + STS/IR datasets
- Tempo: 200 GPU-hours
```


### 8.3 Deployment

```python
# Integração com seu pipeline RAG DHBB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('seu-usuario/serafim-900m-portuguese-ptbr-stsir')
embeddings = model.encode(verbetes, batch_size=32, show_progress_bar=True)
```


### 8.4 Checklist Fase 8

- [ ] Criar model card completo
- [ ] Upload HuggingFace
- [ ] Testar download/inference
- [ ] Publicar paper/blog post
- [ ] Registrar no ORCID/arXiv

**Saída**:

- Modelo público em HuggingFace
- Documentação e reprodução completa

***

## 💰 RESUMO DE CUSTOS

### Opção 1: On-Premises (RTX 4090)

```
Hardware:          $20,000 (2x RTX 4090)
Cooling/Power:     $1,500
Total Fases 3-5:   ~180 GPU-hours
Custo eletricidade: ~$36 (180h × 350W × $0.20/kWh)
─────────────────────────────
CUSTO TOTAL:       ~$21,500
```


### Opção 2: Cloud (Lambda Labs A100)

```
Fase 3 Pretraining:  30 h × $1.99 = $60
Fase 4 STS:          40 h × $1.99 = $80
Fase 5 IR:          100 h × $1.99 = $200
Fase 6-7 Testing:    20 h × $1.99 = $40
─────────────────────────────
CUSTO TOTAL:        ~$380
```


### Opção 3: Cloud (AWS p4d.24xlarge)

```
Fase 3 Pretraining:  8 h × $32.77 = $262
Fase 4 STS:         10 h × $32.77 = $328
Fase 5 IR:          25 h × $32.77 = $819
Fase 6-7 Testing:    5 h × $32.77 = $164
─────────────────────────────
CUSTO TOTAL:        ~$1,573
```


***

## 📈 CRONOGRAMA VISUAL

```
Semana  1  2  3  4  5  6  7  8  9 10 11 12
       ├──┼──┤ Fase 1-2: Prep & Infra
                ├──┼──┤ Fase 3: Pretraining
                     ├──┼──┤ Fase 4: STS
                           ├──┼──┤ Fase 5: IR
                                 ├─┤ Fase 6: Ensemble
                                    ├──┼──┤ Fase 7: Testing
                                           ├─┤ Fase 8: Publish
```


***

## 🎯 RESULTADOS ESPERADOS

### Performance STS

- **Baseline (Serafim-900M-PT)**: 0.87 Spearman
- **Target (novo modelo PT-BR)**: **0.89-0.91** Spearman
- **Ganho**: +2-4% em PT-BR específico


### Performance IR

- **Baseline (Serafim-900M-PT)**: 0.78 nDCG@10
- **Target (novo modelo PT-BR)**: **0.80-0.82** nDCG@10
- **Ganho**: +2-4% em PT-BR específico


### Aplicação DHBB

- **Melhoria em buscas de verbetes**: ~10-15% melhor recall
- **Precisão em queries similaridade**: ~5-10% mais preciso

***

## ✅ PRÓXIMOS PASSOS

1. **Selecionar hardware** (Lambda Labs é recomendado para teste)
2. **Coletar datasets STS/IR** (30-50K pares + 200-500K triplets)
3. **Preparar pipeline de treinamento** (repositório GitHub)
4. **Iniciar Fase 3** (Pretraining contrastivo)

**Tempo total**: 8-12 semanas
**Custo total**: \$380-1,573 (cloud) ou \$21,500 (on-premises)
**ROI**: Modelo específico para PT-BR + publicação academia + melhoria DHBB

Quer que eu detalhe alguma fase específica ou ajude na implementação?
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^2][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^3][^30][^31][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://sol.sbc.org.br/index.php/bracis/article/view/33560

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12420640/37c27f4e-fb21-4ec0-b7a8-d7ab35b4aa4c/rag_pipeline.py

[^3]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12420640/2eaf0178-1411-4e5d-9ac7-332cc091aed2/app_streamlit.py

[^4]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12420640/9ec87794-d575-4b16-9c8d-9f834182a74a/rag_pipeline.py

[^5]: https://arxiv.org/abs/2410.08184

[^6]: https://arxiv.org/abs/2311.04354

[^7]: https://arxiv.org/abs/2505.15080

[^8]: https://arxiv.org/abs/2503.22879

[^9]: https://ieeexplore.ieee.org/document/9857911/

[^10]: https://arxiv.org/abs/2407.04528

[^11]: https://arxiv.org/abs/2502.17728

[^12]: https://ieeexplore.ieee.org/document/10913902/

[^13]: https://arxiv.org/abs/2502.00617

[^14]: https://arxiv.org/abs/2502.16762

[^15]: https://arxiv.org/pdf/2110.05722.pdf

[^16]: http://arxiv.org/pdf/2404.07999.pdf

[^17]: https://arxiv.org/pdf/2203.15556v1.pdf

[^18]: https://arxiv.org/pdf/2209.11055.pdf

[^19]: https://arxiv.org/pdf/2504.03655.pdf

[^20]: https://arxiv.org/pdf/1909.08053.pdf

[^21]: https://arxiv.org/pdf/2404.02258.pdf

[^22]: https://arxiv.org/pdf/2002.11794.pdf

[^23]: https://sbert.net/docs/sentence_transformer/training_overview.html

[^24]: https://huggingface.co/PORTULAN/serafim-900m-portuguese-pt-sentence-encoder

[^25]: https://sbert.net/docs/sentence_transformer/training/examples.html

[^26]: https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/SentenceTransformer.py

[^27]: https://huggingface.co/blog/how-to-train-sentence-transformers

[^28]: https://www.reddit.com/r/MachineLearning/comments/1is0q1a/d_finetuning_modernbert_is_taking_3hrs_2_epochs/

[^29]: https://sbert.net/docs/package_reference/sentence_transformer/training_args.html

[^30]: https://www.runpod.io/blog/llm-fine-tuning-gpu-guide

[^31]: https://euroeval.com/datasets/portuguese/

