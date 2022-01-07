# multiconer
SEMEVAL 2022 MultiCoNER experiments

Training for the EN Track:
```
accelerate launch run_luke_ner_no_trainer.py --model_name_or_path studio-ousia/luke-large --dataset_name multiconer --dataset_config_name NER.en --output_dir ./results-EN --task_name ner --return_entity_level_metrics --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --max_entity_length 840 --max_mention_length 130 --gradient_accumulation_steps 8 --learning_rate 1e-5 --num_train_epochs 30
```

Training for the ZH Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path hfl/chinese-roberta-wwm-ext-large --dataset_name multiconer --dataset_config_name NER.zh --output_dir ./results-ZH --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90 --max_length 256
```

Training for the ES Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path PlanTL-GOB-ES/roberta-large-bne --dataset_name multiconer --dataset_config_name NER.es --output_dir ./results-ES --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the RU Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path sberbank-ai/ruRoberta-large --dataset_name multiconer --dataset_config_name NER.ru --output_dir ./results-RU --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the NL Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path pdelobelle/robbert-v2-dutch-base --dataset_name multiconer --dataset_config_name NER.nl --output_dir ./results-NL --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the TR Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path dbmdz/convbert-base-turkish-mc4-cased --dataset_name multiconer --dataset_config_name NER.tr --output_dir ./results-TR --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the KO Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path klue/roberta-large --dataset_name multiconer --dataset_config_name NER.ko --output_dir ./results-KO --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the FA Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path HooshvareLab/roberta-fa-zwnj-base --dataset_name multiconer --dataset_config_name NER.fa --output_dir ./results-FA --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the DE Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path deepset/gelectra-large --dataset_name multiconer --dataset_config_name NER.de --output_dir ./results-DE --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```
