"""
Prompts para extração de dados de CNH.
"""

CNH_FIELDS = [
    "nome",
    "cpf",
    "data_nascimento",
    "doc_identidade",
    "numero_registro",
    "numero_espelho",
    "validade",
    "categoria",
    "observacoes",
    "primeira_habilitacao",
]

CNH_EXTRACTION_PROMPT = """Analise esta imagem de um documento CNH (Carteira Nacional de Habilitação) brasileira e extraia os seguintes campos.

IMPORTANTE:
- Extraia EXATAMENTE o texto como aparece no documento
- Use o formato DD/MM/AAAA para datas
- Se um campo não estiver visível ou legível, retorne null
- O CPF tem 11 dígitos e pode aparecer como ###.###.###-## ou #########/## (formato novo)
- A categoria pode ser: A, B, AB, C, D, E ou combinações

Retorne APENAS um JSON válido com os seguintes campos:
{
    "nome": "nome completo da pessoa",
    "cpf": "###.###.###-##",
    "data_nascimento": "DD/MM/AAAA",
    "doc_identidade": "número do documento de identidade (RG)",
    "numero_registro": "número de registro da CNH",
    "numero_espelho": "número do espelho",
    "validade": "DD/MM/AAAA",
    "categoria": "categoria da CNH (A, B, AB, etc)",
    "observacoes": "observações ou restrições",
    "primeira_habilitacao": "DD/MM/AAAA"
}

Responda SOMENTE com o JSON, sem explicações ou texto adicional."""
