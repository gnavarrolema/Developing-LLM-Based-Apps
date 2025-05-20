from io import BytesIO
import PyPDF2
import logging


def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    """
    Extract text from a PDF file with enhanced quality validation and error handling.
    
    Parameters
    ----------
    pdf_bytes : BytesIO
        PDF file as bytes object
        
    Returns
    -------
    str
        Extracted text from the PDF
        
    Raises
    ------
    ValueError
        If the PDF cannot be read or does not contain extractable text
    """
    pdf_text = ""
    num_pages = 0
    
    try:
        # Intento principal usando PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        num_pages = len(pdf_reader.pages)
        
        if num_pages == 0:
            raise ValueError("El PDF no contiene páginas.")
            
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
            except Exception as e:
                logging.warning(f"Error al extraer texto de la página {page_num}: {str(e)}")
                continue
                
    except Exception as e:
        # Si falla PyPDF2, podríamos agregar aquí métodos alternativos
        # como pdfminer o pdfplumber en una implementación futura
        raise ValueError(f"Error al leer el archivo PDF: {str(e)}")

    # Validación principal: verificar si se extrajo algún texto
    if not pdf_text:
        raise ValueError("No se encontró texto extraíble en el archivo PDF proporcionado.")
    
    # Validación de calidad: verificar longitud mínima
    if len(pdf_text.split()) < 10:
        logging.warning("El texto extraído del PDF es muy corto o podría estar dañado.")
        
    # Validación de contenido legible (palabras comunes en español o inglés)
    common_words = ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'a', 
                    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for']
    
    has_common_words = any(word.lower() in pdf_text.lower() for word in common_words)
    if not has_common_words:
        logging.warning("El texto extraído puede no ser legible. El formato del PDF podría ser complejo.")
    
    # Procesamiento básico del texto extraído
    # Eliminar espacios en blanco múltiples y normalizar saltos de línea
    import re
    pdf_text = re.sub(r'\s+', ' ', pdf_text)
    pdf_text = pdf_text.strip()
    
    return pdf_text