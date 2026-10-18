"""Advisor engine: ranked ways to save, a before/after projection toward a goal, and the action plan.
Every number is computed in code from src/analytics; an LLM only phrases sentences, which are verified."""
from src.advisor.goals import Goal, clear_goals, load_goal, save_goal
from src.advisor.opportunities import Opportunity, find_opportunities
from src.advisor.plan import Plan, PlanItem, build_plan, outcome_sentences, reproject
from src.advisor.projector import Projection, project

__all__ = ["Goal", "save_goal", "load_goal", "clear_goals", "Opportunity", "find_opportunities",
           "Projection", "project", "Plan", "PlanItem", "build_plan", "outcome_sentences",
           "reproject"]
