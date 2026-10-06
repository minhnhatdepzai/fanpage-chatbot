"""Nhận diện sở thích sáng tác và hướng dẫn hình thức văn học đa truyền thống.

Các hướng dẫn là điểm xuất phát về *hình thức*, không phải khuôn mẫu văn hóa. Khi viết bằng tiếng Việt,
những quy tắc dựa trên âm vị tiếng Hán/Nhật/Hàn chỉ có thể là bản chuyển thể có ý thức.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormGuide:
    name: str
    tradition: str
    instruction: str
    source: str


FORM_GUIDES: dict[str, FormGuide] = {
    "jueju": FormGuide(
        "thơ tuyệt cú (jueju)",
        "Trung Hoa",
        "Bốn dòng cô đọng; tạo chuyển động hình ảnh và ý ở phần cuối. Luật thanh/vần cổ điển phụ thuộc tiếng Hán, "
        "nên bản tiếng Việt chỉ được gọi là chuyển thể nếu không kiểm đủ niêm-luật.",
        "https://en.wikipedia.org/wiki/Jueju",
    ),
    "lushi": FormGuide(
        "thơ luật thi (lüshi)",
        "Trung Hoa",
        "Tám dòng; chú ý các cặp câu giữa có đối ý/đối hình ảnh và mạch khai-triển-chuyển-kết. Không tuyên bố đúng "
        "luật bằng-trắc nguyên ngữ khi đang viết tiếng Việt mà chưa kiểm luật.",
        "https://en.wikipedia.org/wiki/L%C3%BCshi_(poetry)",
    ),
    "ci": FormGuide(
        "từ (cí)",
        "Trung Hoa",
        "Nhịp và độ dài câu biến đổi theo điệu. Nếu người dùng không chỉ định từ bài/điệu cụ thể, viết bản phỏng theo "
        "có câu dài-ngắn giàu nhạc tính, không nhận là phục dựng đúng phổ cổ.",
        "https://en.wikipedia.org/wiki/Ci_(poetry)",
    ),
    "haiku": FormGuide(
        "haiku",
        "Nhật Bản",
        "Ba dòng, trực tiếp và cô đọng, ưu tiên một khoảnh khắc/hình ảnh cụ thể. 5-7-5 trong tiếng Việt là đếm âm "
        "tiết chuyển thể, không hoàn toàn tương đương đơn vị âm của tiếng Nhật.",
        "https://poets.org/glossary/haiku",
    ),
    "tanka": FormGuide(
        "tanka",
        "Nhật Bản",
        "Năm dòng theo nhịp chuyển thể 5-7-5-7-7; ba dòng đầu dựng cảnh/cảm giác, hai dòng sau mở rộng hoặc xoay cảm xúc.",
        "https://poets.org/glossary/tanka",
    ),
    "senryu": FormGuide(
        "senryu",
        "Nhật Bản",
        "Ba dòng ngắn gần cấu trúc haiku nhưng hướng vào hành vi, nghịch lý hoặc nét hài hước của con người; không ép "
        "phải có quý ngữ thiên nhiên.",
        "https://poets.org/glossary/senryu",
    ),
    "haibun": FormGuide(
        "haibun",
        "Nhật Bản",
        "Đan xen văn xuôi cô đọng với một haiku; phần thơ không chỉ lặp lại phần văn mà tạo thêm khoảng vang hoặc góc nhìn.",
        "https://poets.org/glossary/haibun",
    ),
    "sijo": FormGuide(
        "sijo",
        "Hàn Quốc",
        "Thường triển khai trong ba dòng dài có nhịp ngắt giữa dòng: mở chủ đề, phát triển, rồi tạo bước ngoặt và kết. "
        "Bản tiếng Việt ưu tiên cấu trúc ý và nhịp hơn việc giả vờ khớp hoàn toàn âm tiết tiếng Hàn.",
        "https://poets.org/glossary/sijo",
    ),
    "sonnet_shakespearean": FormGuide(
        "sonnet kiểu Anh",
        "châu Âu",
        "Mười bốn dòng: ba khổ bốn dòng phát triển ý và một cặp câu kết tạo chốt/đảo ý; giữ một volta rõ ràng.",
        "https://poets.org/glossary/sonnet",
    ),
    "sonnet_petrarchan": FormGuide(
        "sonnet kiểu Ý",
        "châu Âu",
        "Mười bốn dòng: octave đặt vấn đề, sestet đáp lại hoặc chuyển góc nhìn; đánh dấu volta bằng ý chứ không chỉ "
        "bằng từ nối.",
        "https://poets.org/glossary/sonnet",
    ),
    "villanelle": FormGuide(
        "villanelle",
        "châu Âu",
        "Mười chín dòng gồm năm tercet và một quatrain, dùng hai câu điệp luân phiên rồi hội tụ ở khổ cuối.",
        "https://poets.org/glossary/villanelle",
    ),
    "pantoum": FormGuide(
        "pantoum",
        "Đông Nam Á/châu Âu",
        "Các khổ bốn dòng liên kết bằng việc lặp có chủ ý dòng 2 và 4 thành dòng 1 và 3 của khổ sau; câu lặp nên đổi "
        "nghĩa theo ngữ cảnh, không chỉ sao chép cơ học.",
        "https://poets.org/glossary/pantoum",
    ),
    "ballad": FormGuide(
        "ballad",
        "châu Âu",
        "Kể một câu chuyện bằng các khổ giàu nhịp, cảnh và hành động; có thể dùng điệp khúc nhưng phải giữ tiến triển cốt truyện.",
        "https://poets.org/glossary/ballad",
    ),
    "ode": FormGuide(
        "ode",
        "châu Âu",
        "Tập trung ngợi ca hoặc suy tưởng về một đối tượng, phát triển từ quan sát cụ thể tới ý nghĩa lớn hơn.",
        "https://poets.org/glossary/ode",
    ),
    "elegy": FormGuide(
        "elegy",
        "châu Âu",
        "Giọng suy niệm về mất mát; tránh sáo ngữ, để hình ảnh cụ thể dẫn từ đau buồn tới ghi nhớ hoặc chấp nhận.",
        "https://poets.org/glossary/elegy",
    ),
    "spoken_word": FormGuide(
        "spoken word/slam",
        "châu Mỹ đương đại",
        "Viết cho giọng đọc: nhịp nói, điệp âm/điệp cú, điểm tăng lực và câu kết nghe rõ; không biến thành văn xuôi xuống dòng tùy tiện.",
        "https://poets.org/glossary/spoken-word",
    ),
    "free_verse": FormGuide(
        "thơ tự do",
        "quốc tế hiện đại",
        "Không dùng khuôn vần cố định nhưng mỗi chỗ xuống dòng, khoảng trắng, nhịp và hình ảnh phải có dụng ý.",
        "https://poets.org/glossary/free-verse",
    ),
    "luc_bat": FormGuide(
        "lục bát",
        "Việt Nam",
        "Luân phiên câu sáu và tám tiếng, nối vần lưng/vần chân phù hợp; ưu tiên nhạc tính tự nhiên và kiểm lại số tiếng.",
        "https://vi.wikipedia.org/wiki/L%E1%BB%A5c_b%C3%A1t",
    ),
}

_FORM_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:that ngon )?tu tuyet|\bjueju\b"), "jueju"),
    (re.compile(r"\b(?:that ngon|ngu ngon)?\s*(?:bat cu|luat thi)|\blushi\b"), "lushi"),
    (re.compile(r"\btu khuc\b|\bci poetry\b"), "ci"),
    (re.compile(r"\bhaiku\b"), "haiku"),
    (re.compile(r"\btanka\b"), "tanka"),
    (re.compile(r"\bsenryu\b"), "senryu"),
    (re.compile(r"\bhaibun\b"), "haibun"),
    (re.compile(r"\bsijo\b"), "sijo"),
    (re.compile(r"\bsonnet\b.*\b(?:anh|shakespeare)\b|\bshakespearean sonnet\b"), "sonnet_shakespearean"),
    (re.compile(r"\bsonnet\b.*\b(?:y|italy|petrarch)|\bpetrarchan sonnet\b"), "sonnet_petrarchan"),
    (re.compile(r"\bsonnet\b"), "sonnet_shakespearean"),
    (re.compile(r"\bvillanelle\b"), "villanelle"),
    (re.compile(r"\bpantoum\b"), "pantoum"),
    (re.compile(r"\bballad\b|\btho tu su\b"), "ballad"),
    (re.compile(r"\bode\b|\bkhuc ca ngoi\b"), "ode"),
    (re.compile(r"\belegy\b|\bai ca\b"), "elegy"),
    (re.compile(r"\bspoken word\b|\bslam poetry\b|\btho trinh dien\b"), "spoken_word"),
    (re.compile(r"\btho tu do\b|\bfree verse\b"), "free_verse"),
    (re.compile(r"\bluc bat\b"), "luc_bat"),
]

_TRADITIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btrung (?:quoc|hoa)|\bduong thi\b|\bchinese\b"), "Trung Hoa"),
    (re.compile(r"\bnhat ban\b|\bjapanese\b"), "Nhật Bản"),
    (re.compile(r"\bhan quoc\b|\bkorean\b"), "Hàn Quốc"),
    (re.compile(r"\bviet nam\b|\bvietnamese\b"), "Việt Nam"),
    (re.compile(r"\bmy latin\b|\blatin america\b|\bnam my\b"), "Mỹ Latinh"),
    (re.compile(r"\bhoa ky\b|\bbac my\b|\bamerican\b|\bchau my\b"), "châu Mỹ"),
    (
        re.compile(r"\bchau au\b|\beuropean\b|\b(?:van hoc|phong cach|nuoc)\s+(?:anh|phap|duc|nga|y)\b"),
        "châu Âu",
    ),
]

_TONES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blang man\b|\bromantic\b"), "lãng mạn"),
    (re.compile(r"\bu toi\b|\bdark\b|\bgothic\b"), "u tối"),
    (re.compile(r"\bbuon\b|\bmelanchol"), "buồn, trầm"),
    (re.compile(r"\btoi gian\b|\bminimal"), "tối giản"),
    (re.compile(r"\bhao hung\b|\bhung trang\b|\bepic\b"), "hùng tráng"),
    (re.compile(r"\bcham biem\b|\bsatir"), "châm biếm"),
    (re.compile(r"\bhai huoc\b|\bhumor"), "hài hước"),
    (re.compile(r"\bkinh di\b|\bhorror\b"), "kinh dị"),
    (re.compile(r"\bchua lanh\b|\bhealing\b"), "chữa lành"),
    (re.compile(r"\btriet ly\b|\bphilosoph"), "suy tưởng"),
]

_LANGUAGES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbang tieng anh\b|\bin english\b"), "tiếng Anh"),
    (re.compile(r"\bbang tieng nhat\b|\bin japanese\b"), "tiếng Nhật"),
    (re.compile(r"\bbang tieng han\b|\bin korean\b"), "tiếng Hàn"),
    (re.compile(r"\bbang tieng trung\b|\bin chinese\b"), "tiếng Trung"),
    (re.compile(r"\bbang tieng viet\b|\bin vietnamese\b"), "tiếng Việt"),
]


def detect_creative_profile(normalized_text: str) -> dict[str, object]:
    """Trích các lựa chọn người dùng nói rõ; không suy diễn dân tộc từ tên/chủ đề."""
    form = next((key for pattern, key in _FORM_ALIASES if pattern.search(normalized_text)), None)
    guide = FORM_GUIDES.get(form or "")
    tradition = next((name for pattern, name in _TRADITIONS if pattern.search(normalized_text)), None)
    language = next((name for pattern, name in _LANGUAGES if pattern.search(normalized_text)), None)
    tones = tuple(name for pattern, name in _TONES if pattern.search(normalized_text))[:3]
    return {
        "form": form,
        "tradition": tradition or (guide.tradition if guide else None),
        "language": language,
        "tones": tones,
    }


def creative_guide(form: str | None, tradition: str | None) -> str:
    guide = FORM_GUIDES.get(form or "")
    if guide:
        return f"Hình thức {guide.name} ({guide.tradition}): {guide.instruction}"
    if tradition:
        return (
            f"Cảm hứng {tradition}: chọn một hình thức nhất quán theo yêu cầu cụ thể của người dùng; không dùng "
            "ẩm thực, trang phục, địa danh hoặc nét tính cách dân tộc như ký hiệu trang trí rập khuôn."
        )
    return "Chọn hình thức phù hợp nội dung; nhịp, điểm nhìn và hình ảnh phải nhất quán thay vì pha trộn tùy tiện."
