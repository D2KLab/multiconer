# multiconer
SEMEVAL 2022 MultiCoNER experiments

Training for the EN Track:
```
accelerate launch run_ner_no_trainer_no_fast.py --model_name_or_path microsoft/deberta-v3-large --dataset_name multiconer --dataset_config_name NER.en --output_dir ./results-EN --task_name ner --return_entity_level_metrics --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the ZH Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path hfl/chinese-roberta-wwm-ext-large --dataset_name multiconer --dataset_config_name NER.zh --output_dir ./results-ZH --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90 --max_length 256
```

Training for the ES Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path PlanTL-GOB-ES/roberta-large-bne --dataset_name multiconer --dataset_config_name NER.es --output_dir ./results-ES --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the RU Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path sberbank-ai/ruRoberta-large --dataset_name multiconer --dataset_config_name NER.ru --output_dir ./results-RU --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the NL Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path pdelobelle/robbert-v2-dutch-base --dataset_name multiconer --dataset_config_name NER.nl --output_dir ./results-NL --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the TR Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path dbmdz/convbert-base-turkish-mc4-cased --dataset_name multiconer --dataset_config_name NER.tr --output_dir ./results-TR --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the KO Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path klue/roberta-large --dataset_name multiconer --dataset_config_name NER.ko --output_dir ./results-KO --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the FA Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path HooshvareLab/roberta-fa-zwnj-base --dataset_name multiconer --dataset_config_name NER.fa --output_dir ./results-FA --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the DE Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path deepset/gelectra-large --dataset_name multiconer --dataset_config_name NER.de --output_dir ./results-DE --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the HI Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path neuralspace-reverie/indic-transformers-hi-bert --dataset_name multiconer --dataset_config_name NER.hi --output_dir ./results-HI --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 5e-5 --num_train_epochs 90 --max_length 256
```

Training for the BN Track:
```
accelerate launch run_ner_no_trainer.py --model_name_or_path neuralspace-reverie/indic-transformers-bn-bert --dataset_name multiconer --dataset_config_name NER.bn --output_dir ./results-BN --task_name ner --return_entity_level_metrics --pre_device-train_batch_size 32 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --learning_rate 5e-5 --num_train_epochs 90 --max_length 256
```

Training for the Multi Track, create a pretrained model with all the other languages, and then fine tune it on the Multi Track:
```
accelerate launch run_ner_no_trainer_no_fast.py --model_name_or_path microsoft/mdeberta-v3-base --dataset_name multiconer --dataset_config_name NER.en,NER.zh,NER.es,NER.ru,NER.nl,NER.tr,NER.ko,NER.fa,NER.de,NER.hi,NER.bn --output_dir ./results-multi-pretrain --task_name ner --return_entity_level_metrics --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --learning_rate 3e-5 --num_train_epochs 90 --max_length 256
accelerate launch run_ner_no_trainer_no_fast.py --model_name_or_path ./results-multi-pretrain/best-pretrained-model --dataset_name multiconer --dataset_config_name NER.multi --output_dir ./results-multi --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```

Training for the Mix Track, take the best model trained on the Multi dataset and fine tune it on the Mix Track:
```
accelerate launch run_ner_no_trainer_no_fast.py --model_name_or_path ./results-multi/best-multi-model --dataset_name multiconer --dataset_config_name NER.mix --output_dir ./results-mix --task_name ner --return_entity_level_metrics --per_device_train_batch_size 64 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 --learning_rate 3e-5 --num_train_epochs 90
```
