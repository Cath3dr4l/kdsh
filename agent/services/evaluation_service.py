from models.config import ModelConfig
from services.paper_evaluator import PaperEvaluator
from services.pdf_service import extract_pdf_content
from utils.ai_content_detector_tool import TextDetectionTool
from models.schemas import EvaluationResponse


class EvaluationService:
    def __init__(self, reasoning_config: ModelConfig, critic_config: ModelConfig):
        self.evaluator = PaperEvaluator(reasoning_config, critic_config)
        self.content_detector = TextDetectionTool()

    async def evaluate_pdf(self, pdf_path: str) -> EvaluationResponse:
        content = extract_pdf_content(pdf_path)
        return await self.evaluate_text(content)

    async def evaluate_text(self, content: str) -> EvaluationResponse:
        # Check AI content
        ai_percentage = await self.content_detector._run_async(content)

        self.evaluator.tot.refresh_tree()
        # Get initial evaluation
        decision = await self.evaluator.evaluate_paper(content)
        # Re-evaluate if high AI content detected
        if (
            ai_percentage["average_fake_percentage"] > 15.0
        ) and decision.is_publishable:
            decision = await self.evaluator.evaluate_paper(content, regenerate=True)

        # Get thought tree data
        thought_tree_data = {
            "nodes": [
                {
                    "id": node_id,
                    "content": node.content,
                    "aspect": node.aspect,
                    "evaluation": node.evaluation,
                    "parent": node.parent_id,
                }
                for node_id, node in self.evaluator.tot.all_thoughts.items()
            ]
        }

        return EvaluationResponse(
            is_publishable=decision.is_publishable,
            primary_strengths=decision.primary_strengths,
            critical_weaknesses=decision.critical_weaknesses,
            recommendation=decision.recommendation,
            ai_content_percentage=ai_percentage["average_fake_percentage"],
            thought_tree_data=thought_tree_data,
        )
