"""
Smart Multilingual Post-Processor and Lexical Corrector.
Features:
1. Fast O(1) dictionary validation against 97,000+ unigrams (Gujarati, Hindi, English).
2. Phonetic & Nukta autocorrection (e.g. बड -> बढ़, दाकिला -> दाखिला, उचच -> उच्च).
3. Split-token compound word joiner (e.g. कि सी -> किसी, कर ने -> करने, બો લવા -> બોલવા).
4. English casing and punctuation formatting.
5. Domain Hotwords booster vocabulary.
"""

import re
import os
from pathlib import Path
from typing import List, Set, Dict


class STTPostProcessor:
    def __init__(self, unigram_path: str = "data/lm/unigrams.txt"):
        self.vocab: Set[str] = set()
        self.word_freq: Dict[str, int] = {}
        
        # Load lexicon if exists
        path = Path(unigram_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    w = line.strip().lower()
                    if w:
                        self.vocab.add(w)
                        # Top words have higher frequency rank
                        self.word_freq[w] = max(1, 100000 - idx)
        
        # Common phonetic / compound repair rules for Indic scripts using whitespace lookaround
        self.compound_rules = [
            # Team Specific Voice Commands & Mixed-Script Terms
            (r"(?<!\S)(?:And|એનdas|any\s*desk)\s+બંધ\s+કરો(?!\S)", "AnyDesk બંધ કરો"),
            (r"(?<!\S)(?:And|એનdas)\s*(?:અને|and)\s*Postman(?!\S)", "AnyDesk અને Postman"),
            (r"(?<!\S)Pak\s+ફોટો(?!\S)", "Take a photo"),
            (r"(?<!\S)Please\s+m\s+music(?!\S)", "Play some music"),
            (r"(?<!\S)पाने(?!\S)", "पानी"),
            
            # Gujarati numbers, temperatures & commands
            (r"(?<!\S)છબી\s+ડિગ્રી(?!\S)", "છવ્વીસ ડિગ્રી"),
            (r"(?<!\S)છબીસ(?!\S)", "છવ્વીસ"),
            (r"(?<!\S)એસીનુ(?!\S)", "એસીનું"),
            (r"(?<!\S)તાકમાંન(?!\S)", "તાપમાન"),
            (r"(?<!\S)દિગરી(?!\S)", "ડિગ્રી"),
            (r"(?<!\S)ચોબીસ(?!\S)", "ચોવીસ"),
            (r"(?<!\S)પચીસ(?!\S)", "પચ્ચીસ"),
            (r"(?<!\S)સતાવીસ(?!\S)", "સત્તાવીસ"),
            (r"(?<!\S)અઠાવીસ(?!\S)", "અઠ્ઠાવીસ"),
            (r"(?<!\S)ઓગણત્રીસ(?!\S)", "ઓગણત્રીસ"),
            (r"(?<!\S)લાઇટ\s+બંધ\s+કરો(?!\S)", "લાઇટ બંધ કરો"),
            (r"(?<!\S)લાઇટ\s+ચાલુ\s+કરો(?!\S)", "લાઇટ ચાલુ કરો"),
            (r"(?<!\S)પંખો\s+ચાલુ\s+કરો(?!\S)", "પંખો ચાલુ કરો"),
            (r"(?<!\S)પંખો\s+બંધ\s+કરો(?!\S)", "પંખો બંધ કરો"),
            (r"(?<!\S)જાર\s*પર(?!\S)", "ઝાડ પર"),
            (r"(?<!\S)પાડ\s*તડ(?!\S)", "ફળ તોડે"),
            (r"(?<!\S)ચછઢ\s*ને(?!\S)", "ચઢીને"),
            (r"(?<!\S)છોકરાજાર(?!\S)", "છોકરાઓ ઝાડ"),
            (r"(?<!\S)જાણ\s+વા(?!\S)", "જાણવા"),
            (r"(?<!\S)કર\s+વા(?!\S)", "કરવા"),
            (r"(?<!\S)આવ\s+વા(?!\S)", "આવવા"),
            (r"(?<!\S)અભી\s+નેતા(?!\S)", "અભિનેતા"),
            (r"(?<!\S)ફિલમો(?!\S)", "ફિલ્મો"),
            (r"(?<!\S)ફિલમ(?!\S)", "ફિલ્મ"),
            (r"(?<!\S)નમસ\s*તે(?!\S)", "નમસ્તે"),
            (r"(?<!\S)કે\s+મ\s+છો(?!\S)", "કેમ છો"),
            (r"(?<!\S)મજા\s*મા(?!\S)", "મજામાં"),
            
            # Hindi splits & commands
            (r"(?<!\S)कि\s+सी(?!\S)", "किसी"),
            (r"(?<!\S)कर\s+ने(?!\S)", "करने"),
            (r"(?<!\S)हो\s+ने(?!\S)", "होने"),
            (r"(?<!\S)जा\s+ने(?!\S)", "जाने"),
            (r"(?<!\S)दिखा\s+ओ(?!\S)", "दिखाओ"),
            (r"(?<!\S)बजा\s+ओ(?!\S)", "बजाओ"),
            (r"(?<!\S)उचच\s*शिक्षा(?!\S)", "उच्च शिक्षा"),
            (r"(?<!\S)उचच(?!\S)", "उच्च"),
            (r"(?<!\S)दाकिला(?!\S)", "दाखिला"),
            (r"(?<!\S)बड\s+जाती(?!\S)", "बढ़ जाती"),
            (r"(?<!\S)बड\s+गया(?!\S)", "बढ़ गया"),
            (r"(?<!\S)बड(?!\S)", "बढ़"),
            (r"(?<!\S)छबीस(?!\S)", "छब्बीस"),
            (r"(?<!\S)पचीस(?!\S)", "पच्चीस"),
            (r"(?<!\S)सतईस(?!\S)", "सत्ताईस"),
            (r"(?<!\S)अठाईस(?!\S)", "अट्ठाईस"),
            (r"(?<!\S)डिगरी(?!\S)", "डिग्री"),
            (r"(?<!\S)लाईट\s+बंद\s+करो(?!\S)", "लाइट बंद करो"),
            (r"(?<!\S)लाईट\s+चालू\s+करो(?!\S)", "लाइट चालू करो"),
            (r"(?<!\S)पंखा\s+चालू\s+करो(?!\S)", "पंखा चालू करो"),
            (r"(?<!\S)पंखा\s+बंद\s+करो(?!\S)", "पंखा बंद करो"),
            (r"(?<!\S)गाना\s+बजाओ(?!\S)", "गाना बजाओ"),
            (r"(?<!\S)रास्ता\s+दिखाओ(?!\S)", "रास्ता दिखाओ"),
            (r"(?<!\S)मौसम\s+कैसा\s+है(?!\S)", "मौसम कैसा है"),
            (r"(?<!\S)चंदू\s*के\s*चाचा(?!\S)", "चंदू के चाचा"),
            (r"(?<!\S)चाँदी\s*की\s*चम्मच(?!\S)", "चाँदी की चम्मच"),
            (r"(?<!\S)चटनी\s*चखी(?!\S)", "चटनी चखी"),
            
            # English voice commands & phonetic split repairs
            (r"\bturn\s+ofthe\b", "turn off the"),
            (r"\bturn\s+onthe\b", "turn on the"),
            (r"\bin\s+the\s+rom\b", "in the room"),
            (r"\blikes\s+in\s+the\s+room\b", "lights in the room"),
            (r"\bturn\s+of\b", "turn off"),
            (r"\bswitch\s+of\b", "switch off"),
            (r"\bshut\s+of\b", "shut off"),
            (r"\bpower\s+of\b", "power off"),
            (r"\bcut\s+of\b", "cut off"),
            (r"\bturn\s+on\b", "turn on"),
            (r"\bswitch\s+on\b", "switch on"),
            (r"\blight\s+s\b", "lights"),
            (r"\bthe\s+lites\b", "the lights"),
            (r"\broom\s+lites\b", "room lights"),
            (r"\bdegree\s+celsius\b", "degrees celsius"),
            (r"\bdegrees\s+celcius\b", "degrees celsius"),
            (r"\btwenty\s+six\s+degree\b", "twenty six degrees"),
            (r"\btwenty\s+four\s+degree\b", "twenty four degrees"),
            (r"\btwenty\s+five\s+degree\b", "twenty five degrees"),
            (r"\btwenty\s+seven\s+degree\b", "twenty seven degrees"),
            (r"\btwenty\s+eight\s+degree\b", "twenty eight degrees"),
            (r"\bany\s+desk\b", "AnyDesk"),
            (r"\bpost\s+man\b", "Postman"),
            (r"\bplay\s+som\s+music\b", "play some music"),
            (r"\bplay\s+some\s+muzik\b", "play some music"),
            (r"\bwhat\s+is\s+the\s+time\b", "what is the time"),
            (r"\bhow\s+are\s+you\b", "how are you"),
            (r"\bplease\s+turn\b", "please turn"),
            (r"\b(two|too)\s+(dark|light|normal|auto)\s+mode\b", r"to \2 mode"),
            (r"\b(two|too)\s+(play|start|stop|open|close|turn|switch|set|send|call)\b", r"to \2"),
            (r"\b(ieame|iam)\b", "I am"),
            (r"\b(shwicch|shwich|swich|swith)\b", "switch"),
            (r"\b(kittar|gitar|guiter)\b", "guitar"),
            (r"\b(dark\s+mod|darkmode)\b", "dark mode"),
            (r"\b(light\s+mod|lightmode)\b", "light mode"),
        ]

        # Common English phonetic & acoustic spelling mappings
        self.en_phonetic_fixes = {
            "trun": "turn", "torn": "turn", "tern": "turn",
            "oft": "off", "ov": "of",
            "lite": "light", "lites": "lights", "liht": "light", "lihts": "lights",
            "swich": "switch", "swith": "switch",
            "rom": "room", "roome": "room", "rum": "room",
            "temprature": "temperature", "tempreture": "temperature", "temperatur": "temperature",
            "celcius": "celsius", "selsis": "celsius", "selsious": "celsius",
            "degre": "degree", "degres": "degrees", "digri": "degrees", "digree": "degree",
            "muzik": "music", "musick": "music", "misic": "music",
            "plez": "please", "pleze": "please", "plz": "please", "plse": "please",
            "cal": "call", "col": "call",
            "mesage": "message", "msge": "message", "masage": "message",
            "wether": "weather", "wheather": "weather",
            "meting": "meeting", "meating": "meeting",
            "schedul": "schedule", "shedule": "schedule",
            "tommorow": "tomorrow", "tomorow": "tomorrow",
            "yestarday": "yesterday", "yesturday": "yesterday",
            "offise": "office", "ofice": "office",
            "computr": "computer", "camra": "camera",
            "vedio": "video", "audiyo": "audio",
            "anidesk": "anydesk", "anydesck": "anydesk",
            "posman": "postman", "postmen": "postman",
            "windo": "window", "windos": "windows",
            "skren": "screen", "scren": "screen",
            "speker": "speaker", "speek": "speak",
            "volum": "volume", "valume": "volume",
            "acount": "account", "pasword": "password",
            "googal": "google", "gogle": "google",
            "internat": "internet", "netwrok": "network",
            "learining": "learning", "lerning": "learning",
            "agred": "agreed", "agre": "agree",
            "neerest": "nearest", "nerest": "nearest",
            "erport": "airport", "airpot": "airport",
            "chenai": "chennai", "chenay": "chennai",
            "otumn": "autumn", "autum": "autumn",
            "autmn": "autumn", "yelo": "yellow", "yelow": "yellow",
        }

    def correct_word_levenshtein(self, word: str) -> str:
        """Find closest valid unigram with distance=1 if word is not in vocab."""
        w_lower = word.lower()
        if w_lower in self.vocab or len(w_lower) <= 2:
            return word

        # 1. First check phonetic dictionary
        if w_lower in self.en_phonetic_fixes:
            fix = self.en_phonetic_fixes[w_lower]
            return fix.capitalize() if word.istitle() else (fix.upper() if word.isupper() else fix)

        # 2. Generate candidates with edit distance 1 (deletes, transposes, replaces, inserts)
        splits = [(w_lower[:i], w_lower[i:]) for i in range(len(w_lower) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in "abcdefghijklmnopqrstuvwxyz"]
        inserts = [L + c + R for L, R in splits for c in "abcdefghijklmnopqrstuvwxyz"]

        candidates = set(deletes + transposes + replaces + inserts)
        valid_candidates = [c for c in candidates if c in self.vocab]

        if valid_candidates:
            # Pick highest frequency valid word
            best = max(valid_candidates, key=lambda c: self.word_freq.get(c, 0))
            if word.istitle():
                return best.capitalize()
            elif word.isupper():
                return best.upper()
            return best

        return word

    @staticmethod
    def devanagari_to_gujarati(text: str) -> str:
        """Transliterate Devanagari script to Gujarati script (Unicode offset +0x0180)."""
        out = []
        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                gu_code = code + 0x0180
                # Handle Devanagari Danda (। -> .)
                if code == 0x0964:
                    out.append("।")
                elif 0x0A81 <= gu_code <= 0x0AF1:
                    out.append(chr(gu_code))
                else:
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def gujarati_to_devanagari(text: str) -> str:
        """Transliterate Gujarati script to Devanagari script (Unicode offset -0x0180)."""
        out = []
        for ch in text:
            code = ord(ch)
            if 0x0A80 <= code <= 0x0AFF:
                hi_code = code - 0x0180
                if 0x0901 <= hi_code <= 0x097F:
                    out.append(chr(hi_code))
                else:
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    def enforce_script_consistency(self, text: str, target_lang: str = "auto") -> str:
        """Enforce strict script consistency to eliminate cross-language character leakage."""
        if not text:
            return ""

        gu_count = len(re.findall(r"[\u0A80-\u0AFF]", text))
        hi_count = len(re.findall(r"[\u0900-\u097F]", text))

        if target_lang == "gu":
            # Force 100% Gujarati script: any stray Devanagari is converted to Gujarati
            if hi_count > 0:
                text = self.devanagari_to_gujarati(text)
        elif target_lang == "hi":
            # Force 100% Devanagari script: any stray Gujarati is converted to Devanagari
            if gu_count > 0:
                text = self.gujarati_to_devanagari(text)
        elif target_lang == "auto":
            # In auto mode: lock to dominant script so sentences don't mix scripts
            gu_markers = ["છે", "નથી", "કરો", "મારું", "તમે", "કેમ", "હું", "કરવા", "આવ્યા", "ગયા", "હતા"]
            has_gu_marker = any(m in text for m in gu_markers)
            if gu_count > hi_count or has_gu_marker:
                if hi_count > 0:
                    text = self.devanagari_to_gujarati(text)
            elif hi_count > gu_count:
                if gu_count > 0:
                    text = self.gujarati_to_devanagari(text)

        return text

    def process(self, text: str, target_lang: str = "auto") -> str:
        """Apply full post-processing pipeline on transcribed text."""
        if not text:
            return ""

        import unicodedata
        cleaned = unicodedata.normalize("NFC", text.strip())

        # 1. Enforce script consistency (Zero Hindi leakage when Gujarati is target)
        cleaned = self.enforce_script_consistency(cleaned, target_lang=target_lang)

        # 2. Apply compound joiner & phonetic rules
        for pattern, repl in self.compound_rules:
            cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

        # 3. English word-level Levenshtein snapping for unrecognized English tokens
        words = cleaned.split()
        repaired_words = []
        for w in words:
            if re.match(r"^[A-Za-z]+$", w):
                repaired_words.append(self.correct_word_levenshtein(w))
            else:
                repaired_words.append(w)

        res = " ".join(repaired_words)

        # 4. Collapse multiple spaces
        res = re.sub(r"\s+", " ", res).strip()

        # 5. English sentence capitalization
        if res and res[0].isascii() and res[0].islower():
            res = res[0].upper() + res[1:]

        # 6. Final script consistency pass
        res = self.enforce_script_consistency(res, target_lang=target_lang)

        # 7. Restore punctuation & casing
        res = self.restore_punctuation(res, target_lang=target_lang)

        return res

    def restore_punctuation(self, text: str, target_lang: str = "auto") -> str:
        """Intelligently restore commas, question marks, and terminal periods."""
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text.strip())
        words = text.split()
        if not words:
            return ""

        first_lower = words[0].lower().rstrip(",")
        en_intros = {"please", "kindly", "hello", "hi", "hey", "yes", "no", "okay", "ok"}
        gu_intros = {"નમસ્તે", "નમસ્કાર", "હા", "ના", "કૃપા કરીને"}
        hi_intros = {"नमस्ते", "नमस्कार", "हाँ", "नहीं", "कृपया"}

        if len(words) > 2:
            if first_lower in en_intros and not words[0].endswith(","):
                words[0] = words[0] + ","
            elif words[0] in gu_intros and not words[0].endswith(","):
                words[0] = words[0] + ","
            elif words[0] in hi_intros and not words[0].endswith(","):
                words[0] = words[0] + ","

        formatted = " ".join(words)

        # Capitalize standalone "i" and contractions
        formatted = re.sub(r"\bi\b", "I", formatted)
        formatted = re.sub(r"\bi'm\b", "I'm", formatted)
        formatted = re.sub(r"\bi've\b", "I've", formatted)
        formatted = re.sub(r"\bi'll\b", "I'll", formatted)
        formatted = re.sub(r"\bi'd\b", "I'd", formatted)

        # Capitalize first character if English/ASCII
        if formatted and formatted[0].isascii() and formatted[0].islower():
            formatted = formatted[0].upper() + formatted[1:]

        # Capitalize after sentence-ending punctuation
        def cap_after_punct(match):
            return match.group(1) + match.group(2).upper()
        formatted = re.sub(r"([.?!]\s+)([a-z])", cap_after_punct, formatted)

        # Terminal punctuation (? vs .)
        if not formatted.endswith((".", "?", "!", "।", ";")):
            en_q_words = {
                "what", "when", "where", "why", "who", "whom", "whose", "which", "how",
                "is", "are", "am", "can", "could", "would", "should", "will", "do", "does", "did",
                "have", "has", "had"
            }
            is_en_q = (first_lower in en_q_words)
            hi_q_words = ["क्या", "कहाँ", "कब", "कैसे", "क्यों", "किस", "किसे", "किसका", "कितना", "कितनी", "कितने", "कौन", "कैसा"]
            is_hi_q = any(re.search(rf"(?<!\S){q}(?!\S)", formatted) for q in hi_q_words)
            gu_q_words = ["શું", "કેમ", "ક્યાં", "ક્યારે", "કેવી", "કેવો", "કેવા", "કેવું", "કોણ", "કોને", "કોનું", "કેટલા", "કેટલી", "કેટલું"]
            is_gu_q = any(re.search(rf"(?<!\S){q}(?!\S)", formatted) for q in gu_q_words)

            if is_en_q or is_hi_q or is_gu_q:
                formatted += "?"
            else:
                formatted += "."

        return formatted

