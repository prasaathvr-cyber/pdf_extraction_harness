import json
import re
from pathlib import Path

import fitz

from config.settings import HARNESS_MAX_ITERATIONS, OUTPUT_DIR
from harness_core.types import HarnessState, AgentDecision
from context.context_manager import ContextManager
from tools.registry import ToolRegistry
from memory.session_logger import SessionLogger
from hooks.hook_registry import HookRegistry
from prompts.prompt_builder import PromptBuilder
from models.bedrock_client import BedrockClient

# ← NEW: Add these 2 imports for Phase 4
from agents.validation_agents import SubAgentOrchestrator
from tools.validation_layer import ExtractionValidator


class PDFExtractionHarness:
    """
    Component #1 - The While Loop & Iteration Control
    
    USER-INFORMED GOAL-DRIVEN EXTRACTION:
    User provides a goal (goal parameter), harness figures out the rest.
    
    Now has 4 phases:
    1. Analyze page structure (quick skim)
    2. Get extraction strategy (model decides ORDER based on GOAL)
    3. Extract pages until GOAL is achieved (agent is autonomous in HOW)
    4. Validate extraction (NEW - Quality assurance with sub-agents)
    
    This enables TRUE agency — user informs goal, agent autonomously achieves it.
    """

    # ── STEP 1: Add goal parameter to __init__ ──────────────────
    def __init__(
        self,
        pdf_path    : str,
        goal        : str = "extract_all_data",  # ← NEW: User-informed goal
        registry    : ToolRegistry = None,
        context_mgr : ContextManager = None,
        session_log : SessionLogger = None,
        hook_registry: HookRegistry = None,
        bedrock     : BedrockClient = None
    ):
        self.pdf_path    = pdf_path
        self.pdf_name    = Path(pdf_path).name
        self.goal        = goal  # ← NEW: Store user's goal
        self.registry    = registry
        self.context_mgr = context_mgr
        self.session_log = session_log
        self.hooks       = hook_registry
        self.bedrock     = bedrock

        # Build prompt builder with registered tools
        self.prompt_builder = PromptBuilder(registry.list_descriptions())

        # Initialize state
        self.state = HarnessState(
            pdf_path = pdf_path,
            pdf_name = self.pdf_name
        )
        
        # Store goal in state for logging
        self.state.goal = goal

        # ← NEW: Initialize Phase 4 validators
        self.sub_agent_orchestrator = SubAgentOrchestrator()
        self.extraction_validator = ExtractionValidator()
        self.table_2_smds = []

    # ── PUBLIC ENTRY POINT ─────────────────────────────────────
    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  PDF EXTRACTION HARNESS STARTING (Goal-Driven with Phase 4)")
        print(f"  PDF : {self.pdf_name}")
        print(f"  GOAL: {self._format_goal(self.goal)}")
        print(f"{'='*60}\n")

        # Get total pages
        self.state.total_pages = self._get_total_pages()
        print(f"  Total pages detected: {self.state.total_pages}")

        # ════════════════════════════════════════════════════════════
        # PHASE 1: ANALYZE PAGE STRUCTURE
        # ════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"  PHASE 1: Analyzing page structure...")
        print(f"{'─'*60}\n")
        
        self.state.iteration += 1
        
        # Build system prompt for analysis phase
        system_prompt = self.prompt_builder.build(
            agent_type="vision_extraction",
            extra_context="Phase 1: Analyze document structure"
        )
        
        # First user message: ask model to initiate analysis
        user_msg = (
            f"I have a clinical research PDF: {self.pdf_name}\n"
            f"It has {self.state.total_pages} pages.\n"
            f"My goal is: {self._format_goal(self.goal)}\n"
            f"First, let's analyze the structure of this PDF. "
            f"Use the analyze_pages_overview tool to scan all pages and classify them."
        )
        self.state.messages.append({"role": "user", "content": user_msg})
        
        print(f"  Calling model to initiate analysis...")
        try:
            raw_response = self.bedrock.invoke(system_prompt, self.state.messages)
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            self.state.error_count += 1
            return self._save_output()
        
        self.state.messages.append({"role": "assistant", "content": raw_response})
        
        # Execute analyze_pages_overview tool
        try:
            analysis_result = self.registry.execute("analyze_pages_overview", {
                "pdf_path": self.pdf_path
            })
        except Exception as e:
            print(f"  [ERROR] Page analysis failed: {e}")
            return self._save_output()
        
        if analysis_result.status != "success":
            print(f"  [ERROR] {analysis_result.error_message}")
            return self._save_output()
        
        pages_analysis = analysis_result.data["pages"]
        
        # Feed analysis back to model
        analysis_msg = (
            f"Page analysis complete. Here are the results:\n"
            f"{json.dumps(pages_analysis, indent=2)}"
        )
        self.state.messages.append({"role": "user", "content": analysis_msg})
        
        self.session_log.log(
            self.pdf_name, self.state.iteration, "analyze_pages_overview",
            params={"pdf_path": self.pdf_path, "goal": self.goal},
            result={"status": "success", "pages_analyzed": len(pages_analysis)}
        )

        # ════════════════════════════════════════════════════════════
        # PHASE 2: GET EXTRACTION STRATEGY (Based on Goal)
        # ════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"  PHASE 2: Model decides extraction strategy for goal...")
        print(f"{'─'*60}\n")
        
        self.state.iteration += 1
        
        # ── STEP 2: Update Phase 2 to mention goal ──────────────────
        # Ask model to prioritize extraction based on user's GOAL
        user_msg = (
            f"Now, analyze this page structure and create an extraction strategy.\n"
            f"\n"
            f"USER'S GOAL: {self._format_goal(self.goal)}\n"
            f"\n"
            f"Based on this goal, decide the extraction priority:\n"
            f"- HIGH: Critical for achieving the goal\n"
            f"- MEDIUM: Supporting information for the goal\n"
            f"- LOW: Metadata or supplementary (extract if goal requires completeness)\n"
            f"\n"
            f"For goal '{self.goal}':\n"
            f"{self._get_goal_guidance()}\n"
            f"\n"
            f"Use the get_extraction_priority tool to create your strategy."
        )
        self.state.messages.append({"role": "user", "content": user_msg})
        
        print(f"  Calling model to create extraction strategy...")
        try:
            raw_response = self.bedrock.invoke(system_prompt, self.state.messages)
        except Exception as e:
            print(f"  [ERROR] Model call failed: {e}")
            self.state.error_count += 1
            return self._save_output()
        
        self.state.messages.append({"role": "assistant", "content": raw_response})
        
        # Execute get_extraction_priority tool
        try:
            priority_result = self.registry.execute("get_extraction_priority", {
                "pages_analysis": pages_analysis,
                "pdf_path": self.pdf_path,
                "pdf_name": self.pdf_name
            })
        except Exception as e:
            print(f"  [ERROR] Priority planning failed: {e}")
            return self._save_output()
        
        if priority_result.status != "success":
            print(f"  [ERROR] {priority_result.error_message}")
            return self._save_output()
        
        # Store extraction plan in state
        self.state.extraction_plan = priority_result.data["extraction_plan"]
        
        # Store priority-organized pages for ordered extraction
        self.high_priority_pages = priority_result.data.get("high_priority_pages", [])
        self.medium_priority_pages = priority_result.data.get("medium_priority_pages", [])
        self.low_priority_pages = priority_result.data.get("low_priority_pages", [])
        self.all_pages_to_extract = priority_result.data.get("all_pages", [])
        
        # Feed priority plan back to model
        plan_msg = (
            f"Extraction strategy created for goal: {self._format_goal(self.goal)}\n"
            f"\n"
            f"Strategy: {priority_result.data.get('strategy', 'N/A')}\n"
            f"Rationale: {priority_result.data.get('rationale', 'N/A')}\n"
            f"\n"
            f"HIGH priority pages ({len(self.high_priority_pages)}): {self.high_priority_pages}\n"
            f"MEDIUM priority pages ({len(self.medium_priority_pages)}): {self.medium_priority_pages}\n"
            f"LOW priority pages ({len(self.low_priority_pages)}): {self.low_priority_pages}\n"
            f"\n"
            f"Ready to begin extraction in priority order until goal is achieved."
        )
        self.state.messages.append({"role": "user", "content": plan_msg})
        
        self.session_log.log(
            self.pdf_name, self.state.iteration, "get_extraction_priority",
            params={"pages_analysis_count": len(pages_analysis), "goal": self.goal},
            result={
                "status": "success",
                "high_priority": len(self.high_priority_pages),
                "medium_priority": len(self.medium_priority_pages),
                "low_priority": len(self.low_priority_pages),
                "goal": self.goal
            }
        )

        # ════════════════════════════════════════════════════════════
        # PHASE 3: EXTRACT UNTIL GOAL IS ACHIEVED (Main Loop)
        # ════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"  PHASE 3: Extracting until goal is achieved...")
        print(f"{'─'*60}\n")
        
        while (not self.state.is_done and
               self.state.iteration < HARNESS_MAX_ITERATIONS):

            self.state.iteration += 1
            print(f"\n  [Iteration {self.state.iteration}/{HARNESS_MAX_ITERATIONS}]")

            # ── COMPONENT #2: Context Management ──────────────
            self.state.messages = self.context_mgr.manage(self.state.messages)

            # ── Get next page (HIGH → MEDIUM → LOW) ──
            next_page = self._get_next_priority_page()
            
            if next_page is None:
                # No more pages to extract, but check if goal is satisfied
                print(f"  No more priority pages available.")
                if self._goal_achieved():
                    print(f"  ✓ Goal achieved!")
                    self.state.is_done = True
                    break
                else:
                    print(f"  [WARNING] No more pages, but goal not fully satisfied.")
                    self.state.is_done = True
                    break

            # Get current priority level for this page
            current_priority = self._get_page_priority(next_page)

            # ── COMPONENT #7: Assemble System Prompt ──────────
            progress_ctx = self._build_progress_context()
            system_prompt = self.prompt_builder.build(
                agent_type    = "vision_extraction",
                extra_context = progress_ctx
            )

            # Build user message
            user_msg = self._build_user_message(next_page, current_priority)
            self.state.messages.append({"role": "user", "content": user_msg})

            # Call model to extract this page
            print(f"  Calling model to extract page {next_page}...")
            try:
                raw_response = self.bedrock.invoke(system_prompt, self.state.messages)
            except Exception as e:
                print(f"  [ERROR] Model call failed: {e}")
                self.state.error_count += 1
                if self.state.error_count >= 3:
                    print("  Too many errors, stopping.")
                    break
                continue

            # Add assistant response to history
            self.state.messages.append({"role": "assistant", "content": raw_response})

            # Parse decision
            decision = self._parse_decision(raw_response)
            print(f"  Model decision: action='{decision.action}' | {decision.reasoning[:80]}")

            # ── STEP 3: Check if goal achieved before finishing ───────
            # Handle finish request
            if decision.action == "finish":
                # Check if goal is satisfied
                if self._goal_achieved():
                    print(f"  Model says: Goal achieved. Finishing extraction.")
                    self.state.is_done = True
                    self.session_log.log(
                        self.pdf_name, self.state.iteration, "finish",
                        result={
                            "status": "success",
                            "goal": self.goal,
                            "pages_extracted": len(self.state.extracted_pages),
                            "goal_achieved": True
                        }
                    )
                    break
                else:
                    # Goal not achieved, continue extraction
                    remaining_pages = len(self.all_pages_to_extract) - len(self.state.extracted_pages)
                    print(f"  [INFO] Model requested finish, but goal not fully satisfied.")
                    print(f"  [INFO] Continuing extraction ({remaining_pages} pages remain)...")
                    
                    # Force continuation
                    next_page = self._get_next_priority_page()
                    if next_page:
                        user_msg = (
                            f"Please continue extracting. Goal is not yet achieved.\n"
                            f"Next page: {next_page}\n"
                            f"Use vision_extract_page tool."
                        )
                        self.state.messages.append({"role": "user", "content": user_msg})
                    continue

            # Default action should be vision_extract_page
            if decision.action != "vision_extract_page":
                print(f"  [WARNING] Model chose action '{decision.action}', redirecting to extraction.")
                user_msg = (
                    f"Please extract page {next_page} using the vision_extract_page tool.\n"
                    f"This is needed to achieve the goal: {self._format_goal(self.goal)}"
                )
                self.state.messages.append({"role": "user", "content": user_msg})
                continue

            # ── COMPONENT #8: Pre-tool Lifecycle Hook ──────────
            params = self.hooks.trigger_pre(decision.action, decision.params)
            params['pdf_path'] = self.pdf_path
            params['study_name'] = self.state.pdf_name.replace('.pdf', '')

            # ── COMPONENT #3: Execute Tool via Registry ────────
            try:
                tool_result = self.registry.execute(decision.action, params)
            except (KeyError, PermissionError) as e:
                print(f"  [ERROR] {e}")
                self.session_log.log(
                    self.pdf_name, self.state.iteration, decision.action,
                    params=params, result={"status": "error", "message": str(e)}
                )
                continue

            # ── COMPONENT #8: Post-tool Lifecycle Hook ─────────
            self.hooks.trigger_post(decision.action, tool_result.data)

            # ← NEW: QUALITY CONTROL FOR VISION EXTRACTION
            if decision.action == "vision_extract_page" and tool_result.status == "success":
                extraction_data = tool_result.data
                
                # 1. Repair tables if found
                if extraction_data.get("tables"):
                    for table in extraction_data["tables"]:
                        repair_result = self.registry.execute("repair_table_headers", {
                            "table": table,
                            "pdf_path": self.pdf_path,
                            "page_num": next_page
                        })
                        if repair_result.status == "success":
                            table.update(repair_result.data.get("table", {}))
                            print(f"    ✓ Repaired table headers")
                
                # 2. Check and fix text encoding
                if extraction_data.get("text"):
                    encoding_result = self.registry.execute("detect_encoding_issues", {
                        "text": extraction_data["text"]
                    })
                    if encoding_result.status == "success":
                        if encoding_result.data.get("issues"):
                            extraction_data["text"] = encoding_result.data.get("cleaned_text", extraction_data["text"])
                            print(f"    ✓ Fixed encoding: {encoding_result.data.get('issues')}")
                
                # 3. Validate quality
                quality_result = self.registry.execute("validate_extraction_quality", {
                    "extraction": extraction_data,
                    "page_num": next_page
                })
                
                quality_score = quality_result.data.get("quality_score", 70)
                
                # 4. Store if good, fallback if bad
                if quality_score >= 70:
                    print(f"    ✓ Quality check passed ({quality_score}/100)")
                    self._store_result(decision.action, tool_result)
                else:
                    print(f"    ⚠️  Quality {quality_score} < 70, using PyMuPDF fallback...")
                    fallback_result = self.registry.execute("pymupdf_fallback", {
                        "pdf_path": self.pdf_path,
                        "page_num": next_page
                    })
                    if fallback_result.status == "success":
                        self._store_result("vision_extract_page", fallback_result)
                    else:
                        print(f"    ❌ Fallback also failed, skipping page {next_page}")
            else:
                # For non-vision tools or failures, store normally
                self._store_result(decision.action, tool_result)
            
            # Mark this page as extracted
            if decision.action == "vision_extract_page":
                page_num = decision.params.get("page")
                if page_num:
                    self._mark_page_extracted(page_num)

            # ── COMPONENT #6: Log to Session (persistence) ─────
            self.session_log.log(
                self.pdf_name, self.state.iteration, decision.action,
                params=params, result={"status": tool_result.status, **tool_result.data}
            )

            # Feed tool result back to conversation
            result_msg = (
                f"Page {next_page} extraction complete.\n"
                f"Tool result: {tool_result.status}\n"
                f"{json.dumps(tool_result.data, indent=2)[:1000]}"
            )
            self.state.messages.append({"role": "user", "content": result_msg})

        # ════════════════════════════════════════════════════════════
        # ← NEW: PHASE 4: VALIDATION
        # ════════════════════════════════════════════════════════════
        print(f"\n{'─'*60}")
        print(f"  PHASE 4: Validating extraction quality...")
        print(f"{'─'*60}\n")
        
        self.state.iteration += 1
        
        # Validate entire extraction using sub-agents
        validated_extraction = self._validate_extraction()
        
        # Update state with validation results
        self.state.validation_results = validated_extraction.get("sub_agent_validation", {})
        self.state.quality_score = validated_extraction.get("overall_quality", {}).get("score", 0)
        
        # Use validated data
        self.state.text_chunks = [validated_extraction.get("text", "")]
        self.state.tables = validated_extraction.get("tables", [])
        self.state.figures = validated_extraction.get("figures", [])
        
        print(f"  ✓ Validation complete")
        quality = validated_extraction['overall_quality']['quality']
        score = validated_extraction['overall_quality']['score']
        print(f"  Quality: {quality} ({score}/100)")

        # ── SAVE OUTPUT ────────────────────────────────────────
        return self._save_output()

    # ── PRIVATE HELPERS ────────────────────────────────────────

    # ← NEW: Add this method for Phase 4 validation
    def _validate_extraction(self) -> dict:
        """Phase 4: Validate extraction using sub-agents"""
        
        # Compile current extraction
        extraction = {
            "page": self.state.current_page,
            "text": "\n".join(self.state.text_chunks),
            "tables": self.state.tables,
            "figures": self.state.figures
        }
        
        # Run validation with sub-agents
        print("  Running sub-agent validation...")
        print("    • Text cleaning agent")
        print("    • Figure validation agent")
        print("    • Metadata structure agent")
        
        validated_extraction = self.sub_agent_orchestrator.validate_extraction(
            extraction,
            table_2_smds=self.table_2_smds
        )
        
        # Log validation results
        self.session_log.log(
            self.pdf_name, 
            self.state.iteration, 
            "phase_4_validation",
            result={
                "status": "complete",
                "quality_score": validated_extraction["overall_quality"]["score"],
            }
        )
        
        return validated_extraction

    def _format_goal(self, goal: str) -> str:
        """Format goal for display"""
        goal_descriptions = {
            "extract_all_data": "Extract all data from the PDF (100% coverage)",
            "extract_key_findings": "Extract key findings and methodology",
            "extract_methodology": "Extract methodology section",
            "extract_results": "Extract results and findings",
        }
        return goal_descriptions.get(goal, goal)

    def _get_goal_guidance(self) -> str:
        """Get guidance for agent based on goal"""
        guidance = {
            "extract_all_data": (
                "Since goal is complete extraction:\n"
                "- HIGH: All content pages (methods, results, discussion)\n"
                "- MEDIUM: Supporting content (introduction, analysis)\n"
                "- LOW: Metadata (title, abstract, references)\n"
                "Extract everything to achieve this goal."
            ),
            "extract_key_findings": (
                "Since goal is key findings only:\n"
                "- HIGH: Results, key discussions, methodology\n"
                "- MEDIUM: Introduction, supporting analysis\n"
                "- LOW: Title, abstract, references (can skip)\n"
                "Focus on extracting findings efficiently."
            ),
            "extract_methodology": (
                "Since goal is methodology only:\n"
                "- HIGH: Methods section\n"
                "- MEDIUM: Related discussions, design details\n"
                "- LOW: Everything else (can skip)\n"
                "Focus extraction on methods pages."
            ),
            "extract_results": (
                "Since goal is results only:\n"
                "- HIGH: Results, findings, key data\n"
                "- MEDIUM: Methods (needed for context)\n"
                "- LOW: Discussion, references (can skip)\n"
                "Focus on extracting results and findings."
            ),
        }
        return guidance.get(self.goal, "Adapt extraction strategy to achieve the goal.")

    def _goal_achieved(self) -> bool:
        """
        ── STEP 3: Check if goal has been achieved ──
        Different goals have different success criteria.
        """
        if self.goal == "extract_all_data":
            # All pages must be extracted
            return len(self.state.extracted_pages) == len(self.all_pages_to_extract)
        
        elif self.goal == "extract_key_findings":
            # Must have HIGH priority pages (results + methods)
            return len([p for p in self.high_priority_pages if p in self.state.extracted_pages]) == len(self.high_priority_pages)
        
        elif self.goal == "extract_methodology":
            # Must have methodology pages
            methods_pages = [p for p in self.all_pages_to_extract if self._get_page_type(p) == "methods"]
            return all(p in self.state.extracted_pages for p in methods_pages)
        
        elif self.goal == "extract_results":
            # Must have results pages
            results_pages = [p for p in self.all_pages_to_extract if self._get_page_type(p) == "results"]
            return all(p in self.state.extracted_pages for p in results_pages)
        
        else:
            # Default: extract all for unknown goals
            return len(self.state.extracted_pages) == len(self.all_pages_to_extract)

    def _get_page_type(self, page_num: int) -> str:
        """Get the page type from extraction plan"""
        for item in self.state.extraction_plan:
            if item["page"] == page_num:
                return item.get("page_type", "unknown")
        return "unknown"

    def _get_total_pages(self) -> int:
        try:
            doc = fitz.open(self.pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return 0

    def _get_next_priority_page(self) -> int:
        """Get the next page to extract in priority order: HIGH → MEDIUM → LOW"""
        # Try HIGH priority pages first
        for page in self.high_priority_pages:
            if page not in self.state.extracted_pages:
                return page
        
        # Then MEDIUM priority pages
        for page in self.medium_priority_pages:
            if page not in self.state.extracted_pages:
                return page
        
        # Then LOW priority pages
        for page in self.low_priority_pages:
            if page not in self.state.extracted_pages:
                return page
        
        # No more pages to extract
        return None

    def _get_page_priority(self, page_num: int) -> str:
        """Get the priority level (HIGH/MEDIUM/LOW) for a page"""
        if page_num in self.high_priority_pages:
            return "HIGH"
        elif page_num in self.medium_priority_pages:
            return "MEDIUM"
        elif page_num in self.low_priority_pages:
            return "LOW"
        else:
            return "UNKNOWN"

    def _mark_page_extracted(self, page_num: int):
        """Mark a page as extracted"""
        self.state.extracted_pages.add(page_num)

    def _build_progress_context(self) -> str:
        pages_extracted = len(self.state.extracted_pages)
        pages_remaining = len(self.all_pages_to_extract) - pages_extracted
        
        return (
            f"Goal: {self._format_goal(self.goal)} | "
            f"Pages extracted: {pages_extracted}/{len(self.all_pages_to_extract)} | "
            f"Remaining: {pages_remaining} | "
            f"Tables found: {len(self.state.tables)} | "
            f"Figures found: {len(self.state.figures)} | "
            f"Iteration: {self.state.iteration}"
        )

    def _build_user_message(self, next_page: int, priority: str) -> str:
        """Build user instruction for extraction phase"""
        pages_extracted = len(self.state.extracted_pages)
        pages_remaining = len(self.all_pages_to_extract) - pages_extracted - 1
        
        if self.state.iteration == 3:  # First iteration of Phase 3
            return (
                f"Begin extraction for goal: {self._format_goal(self.goal)}\n"
                f"First page ({priority} priority): {next_page}\n"
                f"Extract it using vision_extract_page tool.\n"
                f"Continue until goal is achieved."
            )
        else:
            return (
                f"Page {next_page} ({priority} priority) is next.\n"
                f"Extract it using vision_extract_page tool.\n"
                f"Goal: {self._format_goal(self.goal)}\n"
                f"Continue until goal is achieved."
            )

    def _parse_decision(self, raw_response: str) -> AgentDecision:
        """Parse the model's response to determine next action"""
        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return AgentDecision(
                    action    = parsed.get('action', 'vision_extract_page'),
                    params    = parsed.get('params', {}),
                    reasoning = parsed.get('reasoning', '')
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        raw_lower = raw_response.lower()
        if 'finish' in raw_lower or 'complete' in raw_lower or 'done' in raw_lower:
            return AgentDecision(action='finish', reasoning='Detected completion request')

        if 'vision_extract_page' in raw_lower:
            match = re.search(r'page\s*(\d+)', raw_lower)
            if match:
                page_num = int(match.group(1))
                return AgentDecision(
                    action='vision_extract_page',
                    params={'page': page_num},
                    reasoning=f'Extract page {page_num}'
                )

        return AgentDecision(action='vision_extract_page', reasoning='Continue extraction')

    def _store_result(self, action: str, tool_result):
        """Store vision extraction results into harness state"""
        if tool_result.status != "success":
            return

        data = tool_result.data

        if action == "vision_extract_page":
            text = data.get("text", "")
            if text and len(text) > 50:
                self.state.text_chunks.append(text)
                print(f"    ✓ Text extracted: {len(text)} chars")

            for table in data.get("tables", []):
                self.state.tables.append({
                    "table_num": table.get("table_num", len(self.state.tables) + 1),
                    "page": data.get("page", 0),
                    "caption": table.get("caption", ""),
                    "headers": table.get("headers", []),
                    "rows": table.get("rows", [])
                })
                print(f"    ✓ Table found: {table.get('caption', 'Unnamed')}")

            for figure in data.get("figures", []):
                self.state.figures.append({
                    "figure_num": figure.get("figure_num", len(self.state.figures) + 1),
                    "page": data.get("page", 0),
                    "caption": figure.get("caption", ""),
                    "description": figure.get("description", ""),
                    "structured_data": figure.get("structured_data", "")
                })
                print(f"    ✓ Figure found: {figure.get('caption', 'Unnamed')}")

        elif action == "validate_json":
            pass

    def _save_output(self) -> dict:
        output = self.state.to_output_dict()
        output["session_file"] = self.session_log.get_path()

        output["extraction_method"] = "Goal-Driven Agent-Based Extraction (with Phase 4 Validation)"
        output["total_iterations"] = self.state.iteration
        output["pages_extracted"] = len(self.state.extracted_pages)
        output["pages_expected"] = self.state.total_pages
        output["goal"] = self.goal
        output["goal_achieved"] = self._goal_achieved()
        output["coverage_percent"] = (len(self.state.extracted_pages) / len(self.all_pages_to_extract) * 100) if self.all_pages_to_extract else 0

        # ← NEW: Add Phase 4 validation results to output
        output["phase_4_validation"] = getattr(self.state, "validation_results", {})
        output["quality_score"] = getattr(self.state, "quality_score", 0)

        output_file = OUTPUT_DIR / f"{self.state.pdf_name.replace('.pdf', '')}_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"  EXTRACTION COMPLETE (with Phase 4 Validation)")
        print(f"  Goal: {self._format_goal(self.goal)}")
        print(f"  Goal Achieved: {self._goal_achieved()}")
        print(f"  Quality Score: {output.get('quality_score', 0)}/100")
        print(f"  Text sections : {len(self.state.text_chunks)}")
        print(f"  Tables found  : {len(self.state.tables)}")
        print(f"  Figures found : {len(self.state.figures)}")
        print(f"  Pages extracted: {len(self.state.extracted_pages)}/{len(self.all_pages_to_extract)}")
        print(f"  Coverage: {output['coverage_percent']:.1f}%")
        print(f"  Iterations    : {self.state.iteration}")
        print(f"  Output saved  : {output_file}")
        print(f"  Session log   : {self.session_log.get_path()}")
        print(f"{'='*60}\n")

        return output