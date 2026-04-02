"""
메인 학습 스크립트 (Hydra 기반)
"""
import os
import json
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from tqdm import tqdm
from llama_cpp import Llama
from datasets import load_from_disk

from qa_engine import QA_Engine
from optimizer import PromptOptimizer
from metrics import calculate_em, calculate_f1
from utils import (
    load_wikipedia_documents,
    format_qa_input,
    save_prompt,
    save_json,
    load_prompt
)


def load_model(cfg: DictConfig) -> Llama:
    """LLM 모델 로드"""
    print(f"📦 Loading model: {cfg.model.filename}")
    
    llm = Llama.from_pretrained(
        repo_id=cfg.model.repo_id,
        filename=cfg.model.filename,
        n_ctx=cfg.model.n_ctx,
        n_gpu_layers=cfg.model.n_gpu_layers,
        verbose=cfg.model.verbose
    )
    
    print("Model loaded successfully")
    return llm


def load_data(cfg: DictConfig):
    """데이터셋 로드"""
    print("Loading datasets...")
    
    # # Wikipedia 문서
    # wiki_docs = load_wikipedia_documents(cfg.data.wikipedia_path)
    # print(f"   → Wikipedia: {len(wiki_docs)} documents")
    
    # KorQuAD 데이터셋 (원본 노트북처럼 로컬 디스크에서 로드)
    dataset_path = Path(cfg.data.train_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_path}"
        )

    dataset = load_from_disk(str(dataset_path))
    print(dataset)
    # DatasetDict 형태라면 지정된 split 사용, 단일 Dataset이면 그대로 사용
    if hasattr(dataset, "keys"):
        split_name = cfg.data.validation_split
        if split_name not in dataset:
            raise KeyError(f"Split '{split_name}' not found in dataset at {dataset_path}")
        split_dataset = dataset[split_name]
    else:
        split_dataset = dataset

    train_limit = min(cfg.data.train_size, len(split_dataset))
    train_data = split_dataset.select(range(train_limit))
    
    print(f"   → Training: {len(train_data)} samples")
    return train_data


def initialize_prompt(cfg: DictConfig) -> str:
    """초기 시스템 프롬프트 생성"""
    # 템플릿 로드
    template = load_prompt(cfg.prompt.template_file)
    
    # 초기 규칙 로드
    with open(cfg.prompt.initial_rules_file, 'r', encoding='utf-8') as f:
        rules_data = yaml.safe_load(f)
    
    rules = rules_data['rules']
    rules_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(rules)])
    
    # 템플릿에 규칙 주입
    return template.format(dynamic_rules=rules_text)


def evaluate(qa_model: QA_Engine, data) -> tuple:
    """
    모델 평가
    
    Returns:
        (avg_em, avg_f1, failures)
    """
    total_em = 0.0
    total_f1 = 0.0
    failures = []
    
    for item in tqdm(data, desc="Evaluating"):
        formatted_prompt = f" Title: {item['title']}\nContext: {item['context']}\nQuestion: {item['question']}"
        answers = item['answers']['text']
        pred = qa_model.predict(formatted_prompt)
        em = calculate_em(pred, answers)
        f1 = calculate_f1(pred, answers)
        
        total_em += em
        total_f1 += f1
        
        if em == 0:
            failures.append({
                    "title": item['title'],
                    "question": item['question'],
                    "context": item['context'],
                    "pred": pred,
                    "gt": answers
                })
    
    avg_em = total_em / len(data)
    avg_f1 = total_f1 / len(data)
    
    return avg_em, avg_f1, failures


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 실행 함수"""
    
    print("="*60)
    print("TextGrad-QA Optimization")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # 출력 디렉토리 설정
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_dir = output_dir / "prompts"
    log_dir = output_dir / "logs"
    prompt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    llm = load_model(cfg)
    
    # 데이터 로드
    # wiki_docs, train_data = load_data(cfg)
    train_data = load_data(cfg)
    # 초기 프롬프트 생성
    initial_prompt = initialize_prompt(cfg)
    print(f"\nInitial prompt loaded ({len(initial_prompt)} chars)")
    
    # QA 엔진 초기화
    qa_model = QA_Engine(llm, initial_prompt)
    
    # Optimizer 초기화
    optimizer = PromptOptimizer(
        qa_engine=qa_model,
        llm_engine=llm,
        sample_ratio=cfg.critic.sample_ratio,
        max_rules=cfg.prompt.max_rules,
        evaluation_principles=cfg.critic.evaluation_principles
    )
    
    # 학습 루프
    best_em = 0.0
    best_prompt = initial_prompt
    consecutive_failures = 0
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Starting optimization for {cfg.optimizer.num_epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(cfg.optimizer.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{cfg.optimizer.num_epochs}")
        print(f"{'='*60}")
        
        # 평가
        avg_em, avg_f1, failures = evaluate(qa_model, train_data)
        
        print(f"\nResults:")
        print(f"   EM:  {avg_em:.4f} ({avg_em*100:.2f}%)")
        print(f"   F1:  {avg_f1:.4f}")
        print(f"   Failures: {len(failures)}/{len(train_data)}")
        
        # 결과 저장
        epoch_result = {
            'epoch': epoch + 1,
            'em': float(avg_em),
            'f1': float(avg_f1),
            'failures': len(failures)
        }
        results.append(epoch_result)
        
        # 프롬프트 저장
        if cfg.logging.save_prompts:
            save_prompt(
                qa_model.system_prompt,
                prompt_dir / f"epoch_{epoch+1:02d}_prompt.txt"
            )
        
        # 실패 케이스 저장
        if cfg.evaluation.log_failures and failures:
            save_json(
                failures[:10],  # 상위 10개만
                log_dir / f"epoch_{epoch+1:02d}_failures.json"
            )
        
        # 최고 성능 갱신
        if avg_em > best_em:
            best_em = avg_em
            best_prompt = qa_model.system_prompt
            consecutive_failures = 0
            
            if cfg.experiment.save_best_prompt:
                save_prompt(best_prompt, output_dir / "best_prompt.txt")
                print(f"   New best EM: {best_em:.4f}")
        else:
            consecutive_failures += 1
            print(f"   No improvement ({consecutive_failures}/{cfg.optimizer.max_consecutive_failures})")
            
            # 롤백
            if avg_em < best_em - cfg.optimizer.rollback_threshold:
                print(f"   🔄 Rollback to best prompt")
                qa_model.update_prompt(best_prompt)
        
        # 조기 종료
        if consecutive_failures >= cfg.optimizer.max_consecutive_failures:
            print(f"\nEarly stopping after {consecutive_failures} failures")
            break
        
        # 최적화 (마지막 에포크 제외)
        if epoch < cfg.optimizer.num_epochs - 1 and failures:
            optimizer.step(failures)
    
    # 최종 결과 저장
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"🏆 Best EM: {best_em:.4f} ({best_em*100:.2f}%)")
    
    summary = {
        'experiment': cfg.experiment.name,
        'timestamp': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'results': results,
        'best_em': float(best_em)
    }
    
    save_json(summary, output_dir / "summary.json")
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
