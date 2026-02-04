#!/usr/bin/env python3
"""
CLI para o pipeline de classificação e extração de documentos.
"""

import argparse
import json
import sys
from pathlib import Path

from doc_pipeline import DocumentPipeline, PipelineResult
from doc_pipeline.config import ExtractorBackend


def format_result(result: PipelineResult, verbose: bool = False) -> str:
    """Formata um resultado para exibição no terminal."""
    lines = []

    if result.file_path:
        lines.append(f"Arquivo: {result.file_path}")

    lines.append(f"Tipo: {result.classification.document_type.value}")
    lines.append(f"Confiança: {result.classification.confidence:.1%}")

    if not result.success:
        lines.append(f"Erro: {result.error}")
        return "\n".join(lines)

    if result.extraction and result.data:
        lines.append("\nDados extraídos:")
        data_dict = result.data.model_dump(exclude_none=True)
        for key, value in data_dict.items():
            lines.append(f"  {key}: {value}")

        if verbose:
            lines.append(f"\nBackend: {result.extraction.backend}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de classificação e extração de documentos BR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Pipeline completo (usa models/classifier.pth por padrão)
  python cli.py documento.jpg

  # Usar EasyOCR ao invés de Qwen (usa menos VRAM)
  python cli.py documento.jpg --backend easy-ocr

  # Apenas classificar
  python cli.py documento.jpg --no-extraction

  # Processar pasta com saída JSON
  python cli.py ./documentos/ --json -o resultados.json

  # Usar modelo em outro caminho
  python cli.py documento.jpg -m /caminho/para/modelo.pth
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Imagem ou pasta para processar",
    )
    parser.add_argument(
        "--modelo",
        "-m",
        type=str,
        default=None,
        help="Caminho do modelo classificador .pth (default: models/classifier.pth)",
    )
    parser.add_argument(
        "--tipo",
        "-t",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b2", "efficientnet_b4"],
        help="Tipo do modelo EfficientNet",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="qwen-vl",
        choices=["qwen-vl", "easy-ocr"],
        help="Backend para extração de dados (easy-ocr usa menos VRAM)",
    )
    parser.add_argument(
        "--classifier-device",
        type=str,
        default=None,
        help="Device para o classificador (cuda:0, cpu)",
    )
    parser.add_argument(
        "--extractor-device",
        type=str,
        default=None,
        help="Device para o extractor (pode ser diferente)",
    )
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Apenas classificar, sem extrair dados",
    )
    parser.add_argument(
        "--min-confianca",
        type=float,
        default=0.5,
        help="Confiança mínima para classificação (0-1)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Saída em formato JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Arquivo de saída para JSON (com --json)",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Usar FP8 no classificador (requer GPU Hopper/Blackwell)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Saída detalhada",
    )

    args = parser.parse_args()

    # Inicializa pipeline
    pipeline = DocumentPipeline(
        classifier_model_path=args.modelo,
        classifier_model_type=args.tipo,
        classifier_device=args.classifier_device,
        classifier_fp8=args.fp8,
        extractor_backend=ExtractorBackend(args.backend),
        extractor_device=args.extractor_device,
    )

    input_path = Path(args.input)
    extract = not args.no_extraction

    if input_path.is_file():
        # Processa arquivo único
        result = pipeline.process(
            input_path,
            extract=extract,
            min_confidence=args.min_confianca,
        )

        if args.json:
            output = result.model_dump(mode="json")
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Resultado salvo em: {args.output}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(format_result(result, verbose=args.verbose))

    elif input_path.is_dir():
        # Processa pasta
        results = list(
            pipeline.process_folder(
                input_path,
                extract=extract,
                min_confidence=args.min_confianca,
            )
        )

        if args.json:
            output = [r.model_dump(mode="json") for r in results]
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Resultados salvos em: {args.output}")
            else:
                print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"\nProcessados: {len(results)} arquivos\n")
            success = sum(1 for r in results if r.success)
            failed = len(results) - success
            print(f"Sucesso: {success} | Falhas: {failed}\n")

            for result in results:
                print("-" * 50)
                print(format_result(result, verbose=args.verbose))

    else:
        print(f"Erro: arquivo ou pasta não encontrado: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
