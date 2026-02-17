# Video Script – CellRepair Health Educator (max. 3 Minuten)

**Format:** Screen Recording + Voiceover (oder Webcam-Bild + Screencast)
**Empfohlenes Tool:** OBS Studio, Loom, oder QuickTime (Mac)
**Zielzeit:** 2:30 – 2:50 Min (Puffer zur 3-Min-Grenze)

---

## [0:00 – 0:15] Intro + Problem (15 Sek.)

**Zeige:** Titelfolie oder Terminal mit Projekt-Name

**Sage (DE oder EN – Jury erwartet EN):**

> "When a doctor says 'oxidative stress' or 'cellular senescence', most patients leave the room more confused than before. Health literacy is a massive gap – and cellular biology is one of the hardest topics to explain to non-experts. We built CellRepair Health Educator to close this gap."

---

## [0:15 – 0:40] Lösung + MedGemma (25 Sek.)

**Zeige:** Architektur-Diagramm (einfach: Patient Question → MedGemma 1.5 → Clear Answer)

**Sage:**

> "CellRepair Health Educator uses MedGemma 1.5 – Google's open medical AI model – as its reasoning backbone. We give it a carefully designed system prompt that instructs it to explain complex cellular health concepts using everyday language, analogies, and actionable lifestyle advice. The 4-billion parameter model runs locally – no cloud, no data sharing – perfect for clinics and patient kiosks."

---

## [0:40 – 2:00] Live Demo (80 Sek.)

**Zeige:** Terminal – Demo starten

```bash
python3 demo_cellrepair_health.py
```

**Sage während das Modell lädt:**

> "Let me show you how it works. We run five realistic patient scenarios covering cellular repair, oxidative stress, lifestyle factors, inflammation, and aging."

**Zeige:** Erste Antwort wird generiert

> "Here's the first scenario: A patient asks 'What does cellular repair actually mean for my body?' Look at MedGemma's response – it explains autophagy as 'your body's recycling program', describes DNA repair in simple terms, and gives practical tips. This is medically grounded but genuinely accessible."

**Zeige:** Scroll durch 2-3 weitere Antworten

> "The oxidative stress scenario – a patient got blood work results they don't understand. MedGemma explains free radicals with a rust analogy and gives concrete dietary recommendations."

> "And here – lifestyle and cell health. Evidence-based advice on how fasting, sleep, and exercise affect cells at the molecular level, explained without jargon."

---

## [2:00 – 2:30] Impact + Deployment (30 Sek.)

**Zeige:** Zusammenfassung / Impact Slide

**Sage:**

> "The real impact: MedGemma's small footprint means this runs on a tablet in a waiting room. Patients arrive better informed for their consultation. No internet needed, no data leaves the device – fully HIPAA and GDPR compatible. Future versions could use MedGemma's multimodal capabilities to explain medical images directly to patients."

---

## [2:30 – 2:50] Abschluss (20 Sek.)

**Zeige:** CellRepair AI Logo / Kontakt

**Sage:**

> "CellRepair Health Educator bridges the gap between molecular biology and patient understanding. Built with MedGemma 1.5, deployable on edge hardware, addressing a real need in preventive healthcare. Thank you."

---

## Produktions-Tipps

1. **Sprache:** Englisch (internationale Jury)
2. **Terminal-Font:** Groß genug dass Text lesbar ist (min. 16pt)
3. **Tempo:** Ruhig und klar sprechen – lieber 2:30 als gehetzt 3:00
4. **Falls MedGemma langsam:** Vorher aufnehmen und schneiden, oder Output vorher generieren und im Video zeigen
5. **Kein CellRepair-internes Branding** (keine ACI, Ultra Beyond Genie etc.) – nur "CellRepair AI"
6. **Disclaimer im Video:** "Research prototype – not for clinical use"
