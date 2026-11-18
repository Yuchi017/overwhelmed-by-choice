import csv
import ast
from typing import Dict, List, Any, Optional
from collections import Counter
from .base_dataset_simple import DDMDataset
from ddm_core import stable_shuffle
import random


class ArxivDataset(DDMDataset):
    # ArXiv paper classification dataset limited to top-K most frequent categories.
    # This makes the classification task more tractable than using all 1177 categories.
    
    def __init__(
        self, 
        dataset_path: str, 
        top_k: int = 175, 
        num_options_per_prompt: Optional[int] = None,
        reasoning=None,
        group=None,
        # none_of_the_above=False
    ):
        """
        Args:
            dataset_path: Path to the ArXiv CSV file
            top_k: Number of most frequent categories to use (default: 20)
            num_options_per_prompt: Number of options to show per prompt.
                                   If None, shows all top-K categories.
                                   If specified, samples (N-1) distractors from top-K + gold label
            reasoning: Optional ReasoningConfig for advanced prompting
        """
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.num_options_per_prompt = num_options_per_prompt
        self.top_categories = None
        # self.none_of_the_above = none_of_the_above
        self.group = group
        super().__init__(reasoning=reasoning, group=group)
    
    def load_data(self) -> List[Dict[str, Any]]:
        # Count category frequencies
        category_counts = Counter()
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                terms = ast.literal_eval(row['terms'])
                if terms:
                    # Count ALL categories (not just primary)
                    # This gives us access to all 1177 categories
                    for term in terms:
                        category_counts[term] += 1
        
        # Get top-K categories
        self.top_categories = [cat for cat, _ in category_counts.most_common(self.top_k)]
        
        print(f"Using top {self.top_k} categories: {self.top_categories}")
        
        # Load only papers with primary category in top-K
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                terms = ast.literal_eval(row['terms'])
                title = row['titles'].strip()
                abstract = row['abstracts'].strip()
                
                # Use papers where ANY of their terms is in top categories
                matching_terms = [t for t in terms if t in self.top_categories]

                if matching_terms:
                    # Deterministically select ONE of the matching terms as the gold label
                    selected_term = stable_shuffle(matching_terms, str(idx), "term_selection")[0]
                    primary_category = selected_term
                    text = f"{title}\n\n{abstract}"
                    
                    # Create options excluding OTHER valid categories for this paper
                    # Only the selected category should appear in options
                    other_valid_categories = [t for t in matching_terms if t != primary_category]
                    available_categories = [c for c in self.top_categories if c not in other_valid_categories]
                    
                    # Determine options for this sample
                    if self.num_options_per_prompt is None:
                        # Use all available categories (excluding other valid ones)
                        options = available_categories
                    else:
                        # Sample (N-1) distractors + gold label from available categories
                        options = self._sample_options(primary_category, available_categories, self.num_options_per_prompt, str(idx))
                    
                    data.append({
                        'id': idx,
                        'text': text,
                        'options': options,
                        'label': primary_category,
                        'meta': {
                            'title': title,
                            'abstract': abstract,
                            'all_terms': terms,
                            'num_terms': len(terms),
                            'total_categories': len(self.top_categories)
                        }
                    })
        
        print(f"Loaded {len(data)} papers with top-{self.top_k} categories")
        if self.num_options_per_prompt:
            print(f"Showing {self.num_options_per_prompt} options per prompt")
        return data
    
    def _sample_options(self, gold_label: str, all_categories: List[str], num_options: int, sample_id: str) -> List[str]:
        """
        Sample options for a single sample: gold label + (N-1) distractors.
        
        Args:
            gold_label: The correct category
            all_categories: All available categories
            num_options: Total number of options to return
            sample_id: Sample ID for deterministic sampling
            
        Returns:
            List of options including gold label and distractors
        """
        # Get distractors (all categories except gold)
        distractors = [cat for cat in all_categories if cat != gold_label]
        
        # Deterministically sample (num_options - 1) distractors
        num_distractors = min(num_options - 1, len(distractors))
        shuffled_distractors = stable_shuffle(distractors, sample_id, "distractor_sample")
        sampled_distractors = shuffled_distractors[:num_distractors]

        # if self.none_of_the_above:
        #     sampled_distractors.append("None of the above")
        
        # Combine gold + distractors (gold will be shuffled into position by base class)
        options = [gold_label] + sampled_distractors
        
        return options
    
    def format_prompt(self, sample: Dict[str, Any], arranged_options: List[str], gold_idx: int) -> str:
        title = sample['meta']['title']
        abstract = sample['meta']['abstract']
        
        # Dataset instruction
        dataset_instruction = "Classify research papers into their primary subject category based on the title and abstract."
        
        class_information = """
        ARXIV SUBJECT CATEGORIES:
        cs.AI (Artificial Intelligence) - also covers I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, I.2.11
        cs.AR (Hardware Architecture) - also covers C.0, C.1, C.5
        cs.CC (Computational Complexity) - also covers F.1, F.2.3, F.4.3, F.2.1, F.2.2
        cs.CE (Computational Engineering, Finance, and Science) - also covers J.2, J.3, J.4
        cs.CG (Computational Geometry) - also covers I.3.5, F.2.2
        cs.CL (Computation and Language) - also covers I.2.7
        cs.CR (Cryptography and Security) - also covers D.4.6, E.3
        cs.CV (Computer Vision and Pattern Recognition) - also covers I.2.10, I.4, I.5
        cs.CY (Computers and Society) - also covers K.0, K.2, K.3, K.4, K.5, K.7
        cs.DB (Databases) - also covers E.2, E.5, H.0, H.2, J.1
        cs.DC (Distributed, Parallel, and Cluster Computing) - also covers C.1.2, C.1.4, C.2.4, D.1.3, D.4.5, D.4.7, E.1
        cs.DL (Digital Libraries) - also covers H.3.5, H.3.6, H.3.7, I.7
        cs.DM (Discrete Mathematics) - also covers G.2, G.3
        cs.DS (Data Structures and Algorithms) - also covers E.1, E.2, F.2.1, F.2.2
        cs.ET (Emerging Technologies)
        cs.FL (Formal Languages and Automata Theory) - also covers F.1.1, F.4.3
        cs.GL (General Literature) - also covers A
        cs.GR (Graphics) - also covers I.3
        cs.GT (Computer Science and Game Theory)
        cs.HC (Human-Computer Interaction) - also covers H.1.2, H.5
        cs.IR (Information Retrieval) - also covers H.3.0, H.3.1, H.3.2, H.3.3, H.3.4
        cs.IT (Information Theory) - also covers E.4, H.1.1
        cs.LG (Machine Learning)
        cs.LO (Logic in Computer Science) - also covers D.2.4, F.3.1, F.4.0, F.4.1, F.4.2, F.4.3
        cs.MA (Multiagent Systems) - also covers I.2.11
        cs.MM (Multimedia) - also covers H.5.1
        cs.MS (Mathematical Software) - also covers G.4
        cs.NA (Numerical Analysis) - also covers G.1
        cs.NE (Neural and Evolutionary Computing) - also covers C.1.3, I.2.6, I.5
        cs.NI (Networking and Internet Architecture) - also covers C.2
        cs.OH (Other Computer Science)
        cs.OS (Operating Systems) - also covers D.4.1, D.4.2, D.4.3, D.4.4, D.4.5, D.4.7, D.4.9
        cs.PF (Performance) - also covers D.4.8, K.6.2
        cs.PL (Programming Languages) - also covers D.1, D.3
        cs.RO (Robotics) - also covers I.2.9
        cs.SC (Symbolic Computation) - also covers I.1
        cs.SD (Sound) - also covers H.5.5, H.1.2, H.5.1, H.5.2, I.2.7, I.5.4, I.6.3, J.5, K.4.2
        cs.SE (Software Engineering) - also covers D.2, D.2.4
        cs.SI (Social and Information Networks) - also covers F.2, G.2, G.3, H.2, I.2, H.3, H.4, H.5, J.1-J.7
        cs.SY (Systems and Control)
        econ.EM (Econometrics)
        econ.GN (General Economics)
        econ.TH (Theoretical Economics)
        eess.AS (Audio and Speech Processing)
        eess.IV (Image and Video Processing)
        eess.SP (Signal Processing)
        eess.SY (Systems and Control)
        math.AC (Commutative Algebra)
        math.AG (Algebraic Geometry)
        math.AP (Analysis of PDEs)
        math.AT (Algebraic Topology)
        math.CA (Classical Analysis and ODEs)
        math.CO (Combinatorics)
        math.CT (Category Theory)
        math.CV (Complex Variables)
        math.DG (Differential Geometry)
        math.DS (Dynamical Systems)
        math.FA (Functional Analysis)
        math.GM (General Mathematics)
        math.GN (General Topology)
        math.GR (Group Theory)
        math.GT (Geometric Topology)
        math.HO (History and Overview)
        math.IT (Information Theory)
        math.KT (K-Theory and Homology)
        math.LO (Logic)
        math.MG (Metric Geometry)
        math.MP (Mathematical Physics)
        math.NA (Numerical Analysis)
        math.NT (Number Theory)
        math.OA (Operator Algebras)
        math.OC (Optimization and Control)
        math.PR (Probability)
        math.QA (Quantum Algebra)
        math.RA (Rings and Algebras)
        math.RT (Representation Theory)
        math.SG (Symplectic Geometry)
        math.SP (Spectral Theory)
        math.ST (Statistics Theory)
        astro-ph.CO (Cosmology and Nongalactic Astrophysics)
        astro-ph.EP (Earth and Planetary Astrophysics)
        astro-ph.GA (Astrophysics of Galaxies)
        astro-ph.HE (High Energy Astrophysical Phenomena)
        astro-ph.IM (Instrumentation and Methods for Astrophysics)
        astro-ph.SR (Solar and Stellar Astrophysics)
        cond-mat.dis-nn (Disordered Systems and Neural Networks)
        cond-mat.mes-hall (Mesoscale and Nanoscale Physics)
        cond-mat.mtrl-sci (Materials Science)
        cond-mat.other (Other Condensed Matter)
        cond-mat.quant-gas (Quantum Gases)
        cond-mat.soft (Soft Condensed Matter)
        cond-mat.stat-mech (Statistical Mechanics)
        cond-mat.str-el (Strongly Correlated Electrons)
        cond-mat.supr-con (Superconductivity)
        gr-qc (General Relativity and Quantum Cosmology)
        hep-ex (High Energy Physics - Experiment)
        hep-lat (High Energy Physics - Lattice)
        hep-ph (High Energy Physics - Phenomenology)
        hep-th (High Energy Physics - Theory)
        math-ph (Mathematical Physics)
        nlin.AO (Adaptation and Self-Organizing Systems)
        nlin.CD (Chaotic Dynamics)
        nlin.CG (Cellular Automata and Lattice Gases)
        nlin.PS (Pattern Formation and Solitons)
        nlin.SI (Exactly Solvable and Integrable Systems)
        nucl-ex (Nuclear Experiment)
        nucl-th (Nuclear Theory)
        physics.acc-ph (Accelerator Physics)
        physics.ao-ph (Atmospheric and Oceanic Physics)
        physics.app-ph (Applied Physics)
        physics.atm-clus (Atomic and Molecular Clusters)
        physics.atom-ph (Atomic Physics)
        physics.bio-ph (Biological Physics)
        physics.chem-ph (Chemical Physics)
        physics.class-ph (Classical Physics)
        physics.comp-ph (Computational Physics)
        physics.data-an (Data Analysis, Statistics and Probability)
        physics.ed-ph (Physics Education)
        physics.flu-dyn (Fluid Dynamics)
        physics.gen-ph (General Physics)
        physics.geo-ph (Geophysics)
        physics.hist-ph (History and Philosophy of Physics)
        physics.ins-det (Instrumentation and Detectors)
        physics.med-ph (Medical Physics)
        physics.optics (Optics)
        physics.plasm-ph (Plasma Physics)
        physics.pop-ph (Popular Physics)
        physics.soc-ph (Physics and Society)
        physics.space-ph (Space Physics)
        quant-ph (Quantum Physics)
        q-bio.BM (Biomolecules)
        q-bio.CB (Cell Behavior)
        q-bio.GN (Genomics)
        q-bio.MN (Molecular Networks)
        q-bio.NC (Neurons and Cognition)
        q-bio.OT (Other Quantitative Biology)
        q-bio.PE (Populations and Evolution)
        q-bio.QM (Quantitative Methods)
        q-bio.SC (Subcellular Processes)
        q-bio.TO (Tissues and Organs)
        q-fin.CP (Computational Finance)
        q-fin.EC (Economics)
        q-fin.GN (General Finance)
        q-fin.MF (Mathematical Finance)
        q-fin.PM (Portfolio Management)
        q-fin.PR (Pricing of Securities)
        q-fin.RM (Risk Management)
        q-fin.ST (Statistical Finance)
        q-fin.TR (Trading and Market Microstructure)
        stat.AP (Applications)
        stat.CO (Computation)
        stat.ME (Methodology)
        stat.ML (Machine Learning)
        stat.OT (Other Statistics)
        stat.TH (Statistics Theory)

        """
        

        dataset_instruction = dataset_instruction + class_information
        dataset_instruction += self._build_prompting_guidance(arranged_options)
        # Query with paper details
        query = f"Paper Title: {title}\n\nAbstract:\n{abstract}\n\nQuestion: What is the primary subject category of this paper?"
        
        # Use standard prompt builder from base class
        return self.build_standard_prompt(dataset_instruction, query, arranged_options)

    from typing import List

    def _build_prompting_guidance(self, arranged_options: List[str]) -> str:
        guidance = []

        # Always-on protocol (this is the part that actually improves accuracy)
        guidance.append(
            "\n\nCLASSIFICATION PROTOCOL (follow strictly):"
            "\nGoal: Choose the SINGLE best category from the provided options."
            "\nHard constraint: Use ONLY evidence from the paper text you were given (title/abstract/keywords/snippet). "
            "Do NOT assume missing details."
            "\n"
            "\nStep 1 — Extract concrete signals (mentally, not necessarily in output):"
            "\n  • Task / problem being solved (what is the paper trying to do?)"
            "\n  • Method / approach (what technique/algorithm/model is central?)"
            "\n  • Domain / application area (where is it applied?)"
            "\n  • Data modality (text/image/audio/time-series/graphs/etc.)"
            "\n  • Named entities: datasets, benchmarks, architectures, losses, evaluation metrics"
            "\n"
            "\nStep 2 — Match options by PRIMARY contribution (not keywords):"
            "\n  Priority order for matching: Task > Method > Domain > Modality."
            "\n  • If the paper introduces a NEW method/architecture: prefer method category over application domain."
            "\n  • If the paper is mainly an application/empirical study: prefer domain/task category."
            "\n  • Ignore superficial keyword overlap unless backed by a concrete signal (e.g., method name, task statement)."
            "\n"
            "\nStep 3 — Quick scoring rubric (internal):"
            "\n  Score each option 0–3:"
            "\n    0 = contradicts / no fit"
            "\n    1 = weak keyword match only"
            "\n    2 = partial fit (some alignment, but not central)"
            "\n    3 = strong fit (directly matches the main contribution)"
            "\n  Select the highest score."
            "\n"
            "\nStep 4 — Tie-break rules (if top scores tie):"
            "\n  • Prefer the option that best describes what is NEW or CENTRAL in the paper."
            "\n  • Prefer more specific over more general (when both are plausible)."
            "\n  • If still tied, choose the option with the strongest explicit evidence (named method/task/dataset)."
            "\n"
            "\nStep 5 — Sanity check against runner-up:"
            "\n  Identify the strongest competing option and verify why it loses based on evidence."
            "\n  If the runner-up has stronger primary-contribution evidence, SWITCH."
            "\n"
            "\nOutput rule:"
            "\n  Return EXACTLY ONE numeric answer in the format <Answer> N </Answer>."
            "\n  N must be the option number from the numbered list above."
            "\n  Do not output the option text or any extra words."
        )

        # Also repeat options in plain text to improve semantic grounding.
        if arranged_options:
            guidance.append(
                "\n\nOPTIONS (for grounding only):\n"
                + "\n".join(f"- {opt}" for opt in arranged_options)
            )

        # Pairwise mode: force direct comparison, no hedging
        if len(arranged_options) == 2:
            guidance.append(
                "\n\nPAIRWISE DECISION MODE:"
                "\n  Compare option 1 vs option 2 ONLY on PRIMARY contribution evidence:"
                "\n    • Which option better matches the task statement?"
                "\n    • Which option better matches the central method/approach?"
                "\n    • Which option better matches the domain/application (if relevant)?"
                "\n  No ties: pick the single better match and output only <Answer> 1 </Answer> or <Answer> 2 </Answer>."
            )

        # CoT mode: encourage multi-step reasoning but keep it internal to prevent verbose outputs
        if getattr(self.reasoning, "enable_cot", False):
            guidance.append(
                "\n\nCoT MODE (internal reasoning only):"
                "\n  Think step-by-step using the protocol above, but DO NOT reveal your full chain-of-thought."
                "\n  Only output the final numeric answer as <Answer> N </Answer>."
            )

        # Self-critique mode: strengthen falsification step
        if getattr(self.reasoning, "enable_self_critique", False):
            guidance.append(
                "\n\nSELF-CRITIQUE MODE (strict):"
                "\n  After selecting an option, attempt to falsify it:"
                "\n    • Find the strongest evidence for the best competing option."
                "\n    • If that evidence better matches the paper’s primary contribution, change your answer."
                "\n  Then output only the final numeric answer as <Answer> N </Answer>."
            )

        return "".join(guidance)