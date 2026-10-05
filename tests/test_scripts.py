from srtljubav.config import YTConfig
from srtljubav.scripts import Script, breaks_at_chars, detect_script, max_chars_for

CFG = YTConfig()


def test_detect_latin():
    assert detect_script("Poštujte i ljubite jedan drugoga.") == Script.LATIN


def test_detect_cyrillic():
    assert detect_script("Поштујте и љубите један другога.") == Script.CYRILLIC


def test_detect_cjk():
    assert detect_script("彼此尊重，彼此相愛。") == Script.CJK


def test_detect_hangul():
    assert detect_script("서로 존경하고 사랑하십시오.") == Script.HANGUL


def test_detect_arabic():
    assert detect_script("احترموا وأحبوا بعضكم البعض") == Script.ARABIC


def test_detect_devanagari():
    assert detect_script("एक दूसरे का आदर और प्रेम करो") == Script.DEVANAGARI


def test_detect_thai():
    assert detect_script("จงเคารพและรักซึ่งกันและกัน") == Script.THAI


def test_mixed_text_picks_dominant():
    # CJK-dominant text with a Latin proper noun
    assert detect_script("保羅在羅馬書 Romans 中說的話語") == Script.CJK


def test_max_chars_per_script():
    assert max_chars_for("Hello world, this is Latin text", CFG) == 42
    assert max_chars_for("彼此尊重，彼此相愛。", CFG) == 18
    assert max_chars_for("จงเคารพและรัก", CFG) == 28
    assert max_chars_for("123 ...", CFG) == 42  # no letters -> OTHER -> default


def test_breaks_at_chars():
    assert breaks_at_chars("彼此尊重") is True
    assert breaks_at_chars("จงเคารพ") is True
    assert breaks_at_chars("Hello world") is False
    assert breaks_at_chars("서로 사랑") is False  # Korean uses spaces
