# בדיקת המודלים החדשים
python -m src --list-models


--save-stage-outputs

# עברית אופטימלית עם הקובץ הקטן
python -m src data/IMG_4225.MP4 --transcription-model ivrit-v2-d4 --force-model

# מהיר ביותר עם הקובץ הבינוני  
python -m src data/IMG_4262.MOV --transcription-model large-v3-turbo

# בדיקת סגמנטציה מותאמת
python -m src data/IMG_4225.MP4 --segment-duration 60 --overlap-duration 15

"Check CLAUDE.md, recent results, and CLI options - what major updates have been added?"

"Check CLAUDE.md, recent results, Recent commits/changes,and CLI options - what major updates have been added?"