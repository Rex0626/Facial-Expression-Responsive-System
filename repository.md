Facial-Expression-Responsive-System/
│
├── README.md
├── setup.md
├── weekly_log.md
├── architecture.md
├── requirements.txt
├── .gitignore
│
├── phase1_expression_detection/
│   │
│   │
│   ├── dataset/
│   │   └── fer2013/
│   │
│   ├── checkpoints/
│   │
│   ├── models/
│   │   └── resnet18_emotion.py
│   │
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── metrics.py
│   │   └── seed.py
│   │
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── test_dataset.py
│   └── test_model.py
│
├── phase2_emotion_memory/
│   │
│   ├── memory_model/
│   ├── database/
│   ├── train_memory.py
│   └── emotion_memory.py
│
├── phase3_response_system/
│   │
│   ├── response_engine.py
│   ├── llm/
│   └── interface/
│
├── experiments/
│
├── docs/
│
└── environment/
    └── environment.yml