"""
Prompts para extração de dados de CIN (Carteira de Identidade Nacional).
"""

CIN_FIELDS = [
    "nome",
    "nome_pai",
    "nome_mae",
    "data_nascimento",
    "naturalidade",
    "cpf",
    "data_expedicao",
    "orgao_expedidor",
]

CIN_EXTRACTION_PROMPT = """Analise esta imagem de um documento CIN (Carteira de Identidade Nacional) brasileira e extraia os seguintes campos.

IMPORTANTE:
- Extraia EXATAMENTE o texto como aparece no documento
- Use o formato DD/MM/AAAA para datas
- Se um campo não estiver visível ou legível, retorne null
- O CPF tem 11 dígitos e pode aparecer como ###.###.###-## ou #########/## (formato novo)
- A CIN é a nova identidade brasileira que substitui o RG — o CPF é o identificador único (não possui número de RG)
- NÃO invente um número de RG; a CIN não possui esse campo

Retorne APENAS um JSON válido com os seguintes campos:
{
    "nome": "nome completo da pessoa",
    "nome_pai": "nome do pai",
    "nome_mae": "nome da mãe",
    "data_nascimento": "DD/MM/AAAA",
    "naturalidade": "cidade/estado de nascimento",
    "cpf": "###.###.###-##",
    "data_expedicao": "DD/MM/AAAA",
    "orgao_expedidor": "órgão expedidor (ex: SSP-SP)"
}

Responda SOMENTE com o JSON, sem explicações ou texto adicional."""
