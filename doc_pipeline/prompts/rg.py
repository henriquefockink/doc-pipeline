"""
Prompts para extração de dados de RG.
"""

RG_FIELDS = [
    "nome",
    "nome_pai",
    "nome_mae",
    "data_nascimento",
    "naturalidade",
    "cpf",
    "rg",
    "data_expedicao",
    "orgao_expedidor",
]

RG_EXTRACTION_PROMPT = """Analise esta imagem de um documento RG (Registro Geral) brasileiro e extraia os seguintes campos.

IMPORTANTE:
- Extraia EXATAMENTE o texto como aparece no documento
- Use o formato DD/MM/AAAA para datas
- Se um campo não estiver visível ou legível, retorne null
- O CPF tem 11 dígitos e pode aparecer como ###.###.###-## ou #########/## (formato novo)
- O RG é o número de registro geral e tem formato variado dependendo do estado
- NÃO confunda CPF com RG: o CPF sempre tem exatamente 11 dígitos

Retorne APENAS um JSON válido com os seguintes campos:
{
    "nome": "nome completo da pessoa",
    "nome_pai": "nome do pai",
    "nome_mae": "nome da mãe",
    "data_nascimento": "DD/MM/AAAA",
    "naturalidade": "cidade/estado de nascimento",
    "cpf": "###.###.###-##",
    "rg": "número do RG",
    "data_expedicao": "DD/MM/AAAA",
    "orgao_expedidor": "órgão expedidor (ex: SSP-SP)"
}

Responda SOMENTE com o JSON, sem explicações ou texto adicional."""
