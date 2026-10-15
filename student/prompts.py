"""ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""
from student import settings


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message
    """
    system_msg = (
        "คุณเป็นผู้เชี่ยวชาญในการทำข้อสอบปรนัย โดยตอบคำถามอย่างแม่นยำและถูกต้อง 100%\n\n"
        "หลักเกณฑ์สำคัญในการวิเคราะห์และตอบ:\n"
        "1. ยึดข้อมูลจาก 'เอกสารอ้างอิง' เป็นหลักเด็ดขาด แม้จะขัดกับความรู้ภายนอก\n"
        "2. อ่านคำถามอย่างละเอียด สังเกตคำปฏิเสธ เช่น 'ไม่', 'ไม่ใช่', 'ยกเว้น', 'ผิด'\n"
        "3. สำหรับโจทย์คำนวณส่วนลดและตัวเลข:\n"
        "   - ตรวจสอบยอดซื้อขั้นต่ำของคูปองแต่ละใบจากตาราง หากยอดซื้อต่ำกว่ายอดขั้นต่ำของคูปองนั้น ส่วนลดจะเป็น 0 บาท\n"
        "   - สำหรับคูปองแบบเปอร์เซ็นต์: คำนวณส่วนลด = ยอดซื้อ * เปอร์เซ็นต์\n"
        "   - สำคัญมาก: หากมี 'ส่วนลดสูงสุดต่อบิล' (เพดานส่วนลด) และส่วนลดที่คำนวณได้มากกว่าเพดานนี้ ลูกค้าจะได้ส่วนลดเท่ากับ 'ส่วนลดสูงสุดต่อบิล' เท่านั้น (เช่น 10% ของ 1,000 คือ 100 บาท แต่เพดานส่วนลดสูงสุดคือ 50 บาท ลูกค้าจะได้ส่วนลด 50 บาท ไม่ใช่ 100 บาท)\n"
        "   - เพดานส่วนลด คือเพดานของเงินส่วนลด ไม่ใช่เพดานยอดซื้อ ยอดซื้อเท่าใดก็ใช้ได้หากถึงขั้นต่ำ\n"
        "4. วิเคราะห์ใน 'reason':\n"
        "   - อ้างอิงข้อความหรือตัวเลขจากเอกสาร\n"
        "   - แสดงขั้นตอนการคำนวณตัวเลขและเปรียบเทียบกับเพดานส่วนลดอย่างชัดเจน\n"
        "   - ตรวจสอบและตัดตัวเลือกที่ไม่ถูกต้อง\n"
        "5. ตอบในรูปแบบ JSON เท่านั้น โดยมี 2 ฟิลด์:\n"
        '   - "reason": สรุปเหตุผลและการคำนวณ\n'
        '   - "answer": ตัวอักษรตัวเลือกที่ถูกต้องตัวเดียวตรงตาม key (เช่น "ก", "ข", "ค", "ง", "จ")'
    )

    context_parts = []
    total_len = 0
    if hits:
        for chunk, score in hits:
            chunk_text = getattr(chunk, "text", str(chunk))
            doc_id = getattr(chunk, "doc_id", "")
            part = f"[เอกสาร: {doc_id}]\n{chunk_text}" if doc_id else chunk_text
            if total_len + len(part) > settings.MAX_CONTEXT_CHARS:
                remaining = settings.MAX_CONTEXT_CHARS - total_len
                if remaining > 100:
                    context_parts.append(part[:remaining])
                break
            context_parts.append(part)
            total_len += len(part)

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีเอกสารอ้างอิงเฉพาะ (ให้ตอบตามหลักวิชาการ)"
    options_str = "\n".join(f"{key}. {text}" for key, text in options)

    user_msg = (
        f"=== เอกสารอ้างอิง ===\n"
        f"{context_str}\n\n"
        f"=== คำถาม ===\n"
        f"{question}\n\n"
        f"=== ตัวเลือก ===\n"
        f"{options_str}\n\n"
        f'กรุณาคิดคำนวณและตรวจสอบเงื่อนไขเพดานส่วนลดให้รอบคอบ แล้วตอบเป็น JSON: {{"reason": "...", "answer": "<key>"}}'
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message
    """
    system_msg = (
        "คุณเป็นสุดยอดโปรแกรมเมอร์ Python ที่เขียนโค้ดได้อย่างถูกต้อง แม่นยำ และรันผ่านทุก unit test\n\n"
        "ข้อกำหนดในการเขียนโค้ด:\n"
        "1. ต้องใช้ signature และชื่อฟังก์ชันตามที่กำหนดอย่างเคร่งครัด\n"
        "2. ใช้เฉพาะ Python standard library เท่านั้น และต้องเขียน import ที่จำเป็น (เช่น `import re`, `import math`) ไว้ที่บรรทัดแรกสุดเสมอ ห้ามลืม import\n"
        "3. ฟังก์ชันต้องคืนค่า (return) ห้าม print() หรือ input()\n"
        "4. หากโจทย์มีข้อห้าม (เช่น ห้ามใช้ max, ห้ามใช้ sorted, ห้ามใช้ eval) ต้องไม่ใช้ฟังก์ชันเหล่านั้นเด็ดขาด\n"
        "5. หากมีเอกสารอ้างอิง:\n"
        "   - ให้ทำตามเงื่อนไขในเอกสารอย่างเคร่งครัด\n"
        "   - การตรวจรูปแบบรหัส: ต้อง `import re` และใช้ regex เช่น `re.fullmatch(r'^BB-[A-Z0-9]{4}-\\d{2}$', code.upper()) is not None`\n"
        "   - การคำนวณส่วนลดตามตารางคูปอง:\n"
        "     * แปลงรหัสเป็นตัวพิมพ์ใหญ่: `code = code.upper()`\n"
        "     * คูปองแต่ละใบมียอดซื้อขั้นต่ำ (min_spend) ของตัวเองในตาราง ห้ามใช้ตัวเลขขั้นต่ำปนกันเด็ดขาด เช่น:\n"
        "       'BB-SAVE-26': min_spend=200, ลด 10% ของ total, เพดานสูงสุด 50\n"
        "       'BB-CAKE-26': min_spend=150, ลด 30 (ห้ามเกิน total), ไม่มีเพดาน\n"
        "       'BB-VIP9-26': min_spend=500, ลด 20% ของ total, เพดานสูงสุด 150\n"
        "     * ตรวจสอบว่า `total >= min_spend` ของคูปองใบนั้นหรือไม่ หาก `total < min_spend` ให้คืน 0.0\n"
        "     * หากรหัสไม่อยู่ในตาราง ให้คืน 0.0\n"
        "     * หากส่วนลดมากกว่าเพดานส่วนลดสูงสุด ให้จำกัดไว้ไม่เกินเพดานสูงสุด\n"
        "     * คืนค่าเป็น float ปัดเศษ 2 ตำแหน่ง (`round(float(discount), 2)`)\n"
        "6. ตรวจสอบชนิดข้อมูลของค่า return ให้ตรงตาม signature (เช่น float คืน float, int คืน int, bool คืน bool)\n"
        "7. ตอบเฉพาะโค้ด Python สมบูรณ์ในบล็อก ```python ... ``` เพียงบล็อกเดียว โดยไม่ต้องมีคำอธิบายอื่น"
    )

    context_parts = []
    total_len = 0
    if hits:
        for chunk, score in hits:
            chunk_text = getattr(chunk, "text", str(chunk))
            doc_id = getattr(chunk, "doc_id", "")
            part = f"[เอกสาร: {doc_id}]\n{chunk_text}" if doc_id else chunk_text
            if total_len + len(part) > settings.MAX_CONTEXT_CHARS:
                remaining = settings.MAX_CONTEXT_CHARS - total_len
                if remaining > 100:
                    context_parts.append(part[:remaining])
                break
            context_parts.append(part)
            total_len += len(part)

    context_section = ""
    if context_parts:
        context_section = (
            f"=== เอกสารอ้างอิงและกติกา ===\n"
            f"{chr(10).join(context_parts)}\n\n"
        )

    sig_line = signature if signature else (f"def {entry_point}(*args, **kwargs):" if entry_point else "")

    user_msg = (
        f"{context_section}"
        f"=== โจทย์ requirement ===\n"
        f"{requirement}\n\n"
        f"=== Function Signature ===\n"
        f"{sig_line}\n\n"
        f"เขียนโค้ดฟังก์ชันให้สมบูรณ์ พร้อม import โมดูลที่จำเป็น ตอบเฉพาะบล็อก ```python ... ```"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
