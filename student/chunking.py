"""ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
"""
import re
from student import settings

HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """แบ่งข้อความตามหัวข้อ markdown โดยเก็บ breadcrumb ของหัวข้อและเนื้อหา"""
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str]] = []
    
    current_levels: dict[int, str] = {}
    current_lines: list[str] = []
    current_breadcrumb = ""
    
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            # ถ้ามีเนื้อหาก่อนหน้า ให้บันทึกเป็น section ก่อน
            if current_lines:
                content = "".join(current_lines).strip()
                if content:
                    sections.append((current_breadcrumb, content))
                current_lines = []
            
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # ล้างระดับที่ลึกกว่า
            current_levels = {k: v for k, v in current_levels.items() if k < level}
            current_levels[level] = title
            
            # สร้าง breadcrumb เช่น "Header 1 > Header 2"
            breadcrumb_parts = [current_levels[k] for k in sorted(current_levels.keys())]
            current_breadcrumb = " > ".join(breadcrumb_parts)
            
            current_lines.append(line)
        else:
            current_lines.append(line)
            
    if current_lines:
        content = "".join(current_lines).strip()
        if content:
            sections.append((current_breadcrumb, content))
            
    return sections


def _split_large_block(header: str, content: str, chunk_size: int, overlap: int) -> list[str]:
    """แบ่ง block ที่ยาวเกิน chunk_size โดยพยายามไม่ตัดกลางตารางหรือ code block"""
    # แยกเป็น paragraphs
    paras = re.split(r"(\n\s*\n)", content)
    
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    header_prefix = f"[{header}]\n" if header and settings.INCLUDE_HEADING else ""
    header_len = len(header_prefix)
    
    effective_limit = max(chunk_size - header_len, 200)
    
    for part in paras:
        if not part:
            continue
        part_len = len(part)
        
        if current_len + part_len <= effective_limit:
            current_parts.append(part)
            current_len += part_len
        else:
            if current_parts:
                full_chunk = header_prefix + "".join(current_parts).strip()
                if full_chunk.strip():
                    chunks.append(full_chunk)
                
                # ทำ overlap
                overlap_text = "".join(current_parts)[-overlap:] if overlap > 0 else ""
                current_parts = [overlap_text, part] if overlap_text else [part]
                current_len = len("".join(current_parts))
            else:
                # กรณี paragraph เดียว ยาวกว่า effective_limit
                # ตัดตามบรรทัด
                sub_lines = part.splitlines(keepends=True)
                for line in sub_lines:
                    if current_len + len(line) <= effective_limit:
                        current_parts.append(line)
                        current_len += len(line)
                    else:
                        if current_parts:
                            full_chunk = header_prefix + "".join(current_parts).strip()
                            if full_chunk.strip():
                                chunks.append(full_chunk)
                            current_parts = [line]
                            current_len = len(line)
                        else:
                            # บรรทัดเดียวยาวมากเกินไป
                            chunks.append(header_prefix + line.strip())
                            current_parts = []
                            current_len = 0

    if current_parts:
        full_chunk = header_prefix + "".join(current_parts).strip()
        if full_chunk.strip():
            chunks.append(full_chunk)
            
    return chunks


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ
    
    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)
    """
    if not text or not text.strip():
        return [doc_id]
        
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    
    if settings.SPLIT_BY_HEADING:
        sections = _split_into_sections(text)
    else:
        sections = [(doc_id, text)]
        
    if not sections:
        sections = [(doc_id, text)]
        
    all_chunks: list[str] = []
    for breadcrumb, content in sections:
        header = breadcrumb if breadcrumb else doc_id
        if len(content) <= chunk_size:
            header_prefix = f"[{header}]\n" if header and settings.INCLUDE_HEADING else ""
            chunk_text = header_prefix + content
            if chunk_text.strip():
                all_chunks.append(chunk_text.strip())
        else:
            sub_chunks = _split_large_block(header, content, chunk_size, overlap)
            all_chunks.extend(sub_chunks)
            
    # ป้องกัน list ว่าง
    cleaned_chunks = [c.strip() for c in all_chunks if c.strip()]
    return cleaned_chunks if cleaned_chunks else [text.strip()]
