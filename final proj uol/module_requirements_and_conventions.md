# CM3070 Final Project - Requirements, Marking & Conventions

Distilled from the module's own teaching material (lecture transcripts + syllabus PDF) so the
key requirements are in one place. Source files in this folder:
`CM3070 Computer Science Final Project 2025 syllabus.pdf` and
`uni sources for final report marking grading etc.docx`.

> Note: our module template is **CM3020 Artificial Intelligence - Project Idea 1 (Orchestrating
> AI models)**. CM3070 is the umbrella final-project module these materials describe.

---

## 1. The process: Four Ds + a silent T

- **Discovery** (≈ wk 1-6): domain + users + requirements + background research; choose template;
  proposal; literature review.
- **Design** (≈ wk 6-11): architecture, components, algorithms, UX flow, work plan (Gantt,
  milestones, critical path), evaluation plan; first prototypes to test feasibility.
- **Development** (≈ wk 11-20): build the software; programming + testing as one iterative loop;
  write up the implementation as you go.
- **Delivery** (≈ wk 20-25): final prototype others can run, final report, summative evaluation,
  demo video.
- **Testing (silent T)** runs through *all* phases (see section 4).

The process is **circular**: phases can loop back (a prototype can send you back to the
literature). But you progress through them in order; you can't start in the middle.

## 2. Assessment structure and weights (from syllabus)

| Item | Week | Weight |
|---|---|---|
| Project Proposal (formative, video) | ~4 | 0% |
| **Preliminary Report + first prototype** | ~10 | **10%** |
| Check-in quizzes | 1-20 | 5% |
| Draft report (formative) | ~18 | 0% |
| Written exam | ~21-22 | 20% |
| Demo video (3-5 min, MP4) | ~24 | 5% |
| **Final report + code** | ~24 | **60%** |

- Formative submissions don't score but give tutor feedback - **do them anyway**.
- The **exam** checks you understand your own process and choices, and can discuss alternatives
  you rejected. Keep notes on *why* each decision was made (our git history + TODO rationale
  already capture much of this).

## 3. Report structure

**Preliminary report (this submission):** 4 chapters - Introduction (1000w), Literature Review
(2500w), Design (2000w), Feature Prototype (1500w); 6000w overall cap. Plus a 3-5 min MP4 demo.

**Final report (later):** **6 chapters**, no per-chapter limits:
1. Introduction - concept + motivation + which template.
2. Literature Review - revised from the 2nd peer review.
3. Design - revised from the 3rd peer review (architecture, tech, work plan, evaluation plan).
4. Implementation - major algorithms/techniques, key code, visual results (screenshots/graphs);
   expanded from the Topic 6 style.
5. Evaluation - the evaluation carried out + results; critical evaluation of the whole project:
   successes, failures, limitations, extensions.
6. Conclusion - summary + broader themes + further work.

`final_report.md` in the repo root already follows these six chapters.

## 4. Testing vs Evaluation (they are different)

- **Testing** = making sure something works as it should: unit tests, day-to-day checks,
  integration. Mechanical correctness. Happens continuously.
- **Evaluation** = bigger/broader: *did the project achieve its original aims, and how well?*
  A judgement, not a pass/fail check.
- In **Discovery**, "testing" means: does the background research support the project idea?
- The **final evaluation is summative**: shift from "how do I improve this?" to "how well did
  it meet the aims, and what did I learn?"
- Choose evaluation methods that fit the project type (ML vs web dev differ); unit testing alone
  is not always the appropriate evaluation.
- **Be honest about failures** - markers are suspicious of "I achieved exactly what I planned".
  Highlighting limitations *raises* the evaluation mark. (We already do this.)
- User testing: use consenting healthy adults as a proxy if the real target group is sensitive.

## 5. Report-writing conventions (house style for the final report)

- Write for a reader who **knows CS but has not seen your work** - show them what you did and
  what's unique. Curator mindset: include only what's needed; cutting saves word count.
- **Past / completed tense** - "the system does / was built", not "I will build". (Future work
  is the exception.)
- **Minimal first person**; focus on what was done, not an "I/me" personal story. (NB: the demo
  *video* is deliberately more personal - that's fine, video != report.)
- **Avoid the passive voice**, especially in the literature review (it hides whose view is
  whose).
- **Justify every claim/decision with evidence.** Proper referencing and citation throughout.
- **Critical evaluation** = judgement in the context of *your* aims (what gap is unfilled? does a
  technique work or not?), not summary or marketing blurb; include real technical detail.
- **Include the journey** - decisions, iterations, things that didn't work - not just final
  results. (Our development log captures this.)
- Techniques that help: read it aloud; explain it to a non-expert; "prototype the report" by
  first writing what each section should *argue*; get others to read it; write a little often.

## 6. Recurring marking criteria

- Knowledge of the area and previous work.
- Critical evaluation of previous work (gaps, judgements tied to aims).
- Concept justified by domain and users.
- Work plan feasible, with contingency.
- Evaluation strategy appropriate to the aims; results well presented and critically analysed.
- Originality = bonus only; most projects score 0 here and that is fine.

## 7. Ethics

- A mandatory ethics quiz exists in the module.
- Avoid: children, vulnerable adults, medical patients, animals; be careful with copyrighted
  material and personal data. UK research-ethics / data-protection norms apply regardless of
  country.
- **Our project's ethics angle**: footage shows identifiable athletes (personal data / likeness),
  and the footage is third-party copyrighted video used under limited educational fair use. The
  final report has an Ethics subsection in the Design chapter covering this. No naming/biometric
  ID of individuals; consent + data-handling policy needed only if named per-fencer profiles are
  ever stored.

## 8. General advice

- Choose something you're passionate about (you live with it ~6 months) - done.
- Start early, progress steadily, don't cram at the end.
- Use tutors and **every feedback opportunity** (proposal + draft report).
- Expect to refine scope; over-scoping is normal (agile exists to cope with it). Don't change
  topic after the proposal stage.
- Keep a journal of decisions and dead-ends (our git history + `final_report.md` dev log + TODO
  serve this purpose).
