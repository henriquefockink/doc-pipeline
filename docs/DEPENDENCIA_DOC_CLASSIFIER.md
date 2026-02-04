# Dependência doc-classifier: Problema e Solução

## Status: RESOLVIDO

A dependência externa foi removida. O código de inferência do `doc-classifier` foi copiado
para `doc_pipeline/classifier/classificar.py`. O projeto agora é 100% portátil.

---

## Contexto Original

O `doc-pipeline` dependia do pacote `doc-classifier` para classificar documentos (RG/CNH). A dependência era resolvida via **volume mount** no Docker:

```yaml
# docker-compose.yml (ANTES - não mais necessário)
volumes:
  - ../doc-classifier:/app/doc-classifier:ro
```

Isso significava que o código do `doc-classifier` precisava estar em uma **pasta irmã** do `doc-pipeline`:

```
tools/
├── doc-pipeline/      ← este projeto
└── doc-classifier/    ← dependência (precisava existir aqui)
```

## Problema (Resolvido)

Ao migrar o `doc-pipeline` para outra máquina ou ambiente:

1. O `git clone` do `doc-pipeline` **não trazia** o `doc-classifier`
2. Os workers falhavam com erro: `Directory '/app/doc-classifier' is not installable`
3. Era necessário clonar manualmente o `doc-classifier` na pasta correta

Isso quebrava a portabilidade e dificultava deploys automatizados.

## Solução Implementada

Copiamos apenas o código de **inferência** (~170 linhas) do `doc-classifier` para dentro do `doc-pipeline`:

```
doc_pipeline/classifier/
├── __init__.py       # Exporta ClassifierAdapter
├── adapter.py        # Wrapper (interface do pipeline)
└── classificar.py    # Código de inferência (copiado do doc-classifier)
```

**Vantagens:**
- Projeto 100% portátil (`git clone` traz tudo)
- Não precisa mais do volume mount
- Código de treino permanece no repo original
- Zero overhead de manutenção (código de inferência raramente muda)

## Soluções Possíveis

### Opção 1: Git Submodule (Recomendada)

Adicionar `doc-classifier` como submódulo do `doc-pipeline`:

```bash
# Adicionar submódulo
cd doc-pipeline
git submodule add <url-do-doc-classifier> vendor/doc-classifier

# Clone passa a trazer tudo junto
git clone --recursive <url-do-doc-pipeline>
```

**Prós:**
- Clone único traz ambos os projetos
- Versão do `doc-classifier` fica "travada" (reprodutibilidade)
- Não precisa publicar pacote

**Contras:**
- Submodules têm curva de aprendizado
- Atualizar versão requer `git submodule update`
- Desenvolvedores precisam lembrar do `--recursive`

**Esforço:** Baixo (algumas horas)

---

### Opção 2: Copiar código para o Dockerfile

Copiar o `doc-classifier` diretamente na build da imagem:

```dockerfile
# Dockerfile.worker-docid
COPY vendor/doc-classifier /app/doc-classifier
RUN pip install /app/doc-classifier
```

**Prós:**
- Imagem Docker é 100% autocontida
- Simples de implementar

**Contras:**
- Código duplicado (precisa copiar manualmente quando atualizar)
- Fácil ficar desatualizado
- Aumenta tamanho do repositório

**Esforço:** Muito baixo (1-2 horas)

---

### Opção 3: Publicar no PyPI (ou registry privado)

Publicar `doc-classifier` como pacote Python:

```dockerfile
# Dockerfile.worker-docid
RUN pip install doc-classifier  # ou do registry privado
```

**Prós:**
- Forma "correta" de distribuir pacotes Python
- Versionamento semântico
- Fácil de usar em outros projetos

**Contras:**
- Precisa configurar CI/CD para publicar
- Overhead de manutenção do pacote
- Se privado, precisa de registry (ex: AWS CodeArtifact, GCP Artifact Registry)

**Esforço:** Médio (1-2 dias para setup inicial)

---

### Opção 4: Monorepo

Mover `doc-classifier` para dentro do `doc-pipeline`:

```
doc-pipeline/
├── api.py
├── workers/
└── packages/
    └── doc-classifier/   ← agora está dentro
```

**Prós:**
- Tudo em um lugar só
- Sem dependências externas
- Refatorações são atômicas

**Contras:**
- Reestruturação significativa
- Se `doc-classifier` for usado por outros projetos, complica

**Esforço:** Médio-Alto (depende do tamanho)

---

## Recomendação

Para o cenário atual, sugiro a **Opção 1 (Git Submodule)** porque:

1. Menor esforço de implementação
2. Mantém os projetos separados
3. Garante reprodutibilidade (versão travada)
4. Não requer infraestrutura adicional (registry, CI/CD)

### Contexto Adicional

O `doc-classifier` contém:
- Código de **treino** de modelos (experimentos, datasets, notebooks)
- Código de **inferência** (usado pelo `doc-pipeline`)

Faz sentido manter separado porque:
- São ciclos de vida diferentes (treino é esporádico, inferência é produção)
- Equipes/pessoas diferentes podem trabalhar em cada um
- O repo de treino pode ter arquivos grandes (datasets, checkpoints)

Com submodule, o `doc-pipeline` usa apenas o código de inferência, apontando para uma versão específica. Quando um novo modelo for treinado e validado, basta atualizar o submodule:

```bash
cd doc-pipeline/vendor/doc-classifier
git pull origin main
cd ..
git add vendor/doc-classifier
git commit -m "chore: update doc-classifier to latest version"
```

## Implementação (Opção 1)

Se aprovado, os passos seriam:

```bash
# 1. Adicionar submodule
cd doc-pipeline
git submodule add git@github.com:org/doc-classifier.git vendor/doc-classifier

# 2. Atualizar docker-compose.yml
# De: ../doc-classifier:/app/doc-classifier:ro
# Para: ./vendor/doc-classifier:/app/doc-classifier:ro

# 3. Commit
git add .gitmodules vendor/doc-classifier docker-compose.yml
git commit -m "feat: add doc-classifier as git submodule"

# 4. Documentar no README
# git clone --recursive <url>
# ou
# git clone <url> && git submodule update --init
```

Tempo estimado: 2-4 horas (incluindo testes).
