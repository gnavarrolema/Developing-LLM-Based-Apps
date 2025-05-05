from io import BytesIO

import PyPDF2


def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    """
    Extract text from a PDF file with quality validation.
    """
    pdf_text = ""
    num_pages = 0
    
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        num_pages = len(pdf_reader.pages)
        
        if num_pages == 0:
            raise ValueError("El PDF no contiene páginas.")
            
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
                
    except Exception as e:
        raise ValueError(f"Error al leer el archivo PDF: {str(e)}")

    # Validar la calidad del texto extraído
    if not pdf_text:
        raise ValueError("No se encontró texto en el archivo PDF proporcionado.")
    
    # Validar que el texto tenga un mínimo de contenido significativo
    if len(pdf_text.split()) < 10:
        raise ValueError("El texto extraído del PDF es demasiado corto o puede estar dañado.")
        
    # Validar contenido legible (presencia de palabras comunes en español o inglés)
    common_words = ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'a', 
                    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for']
    
    has_common_words = any(word.lower() in pdf_text.lower() for word in common_words)
    if not has_common_words:
        raise ValueError("El texto extraído puede no ser legible. Verifica el formato del PDF.")
    
    # Informar estadísticas básicas
    word_count = len(pdf_text.split())
    stats = {
        "num_pages": num_pages,
        "word_count": word_count,
        "chars_count": len(pdf_text)
    }
    
    return pdf_text