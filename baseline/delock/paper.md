




Suning Huang  
Jiaqi Shao  
Ke Wang  
Qianzhong Chen  
Jiankai Sun  
Yanjiang Guo  

Mac Schwager†\dagger  
Jeannette Bohg†\dagger 

Stanford University




Abstract
Have you ever post-trained a generalist vision-language-action (VLA) policy on a small demonstration dataset, only to find that it stops responding to new instructions and is limited to behaviors observed during post-training? We identify this phenomenon as lock-in: after low-data, supervised fine-tuning (SFT), the policy becomes overly specialized to the post-training data and fails to generalize to novel instructions, manifesting as concept lock-in (fixation on training objects/attributes) and spatial lock-in (fixation on training spatial targets). Many existing remedies introduce additional supervision signals, such as those derived from foundation models or auxiliary objectives, or rely on augmented datasets to recover generalization. In this paper, we show that the policy’s internal pre-trained knowledge is sufficient: DeLock mitigates lock-in by preserving visual grounding during post-training and applying test-time contrastive prompt guidance to steer the policy’s denoising dynamics according to novel instructions. Across eight simulation and real-world evaluations, DeLock consistently outperforms strong baselines and matches or exceeds the performance of a state-of-the-art generalist policy post-trained with substantially more curated demonstrations. Experimental videos are available at DeLock.

00footnotetext: †\dagger Equal advising. For any questions, please contact: suning@stanford.edu


Keywords: Robot Foundation Models, Low-Data Post-Training, Contrastive Prompt Guidance




1 Introduction

Generalist robot policies (e.g., vision-language-action models, VLAs) have recently shown impressive generalization capabilities: they can perform a range of tasks in unseen environments and generalize across scene configurations, objects, and open-vocabulary language instructions [23, 28, 42, 29, 4, 44]. Despite this breadth, generalist policies often must be adapted to perform effectively on downstream tasks or a specific robot embodiment, most commonly by post-training on tens to hundreds of hours of curated demonstrations for the target domain [49, 8, 5, 18, 17]. While prior work has shown that such post-training can yield robust policies, collecting large-scale demonstration corpora is expensive and often impractical. Consequently, in real deployments, these generalist policies are frequently post-trained in a low-data regime (i.e., with limited demonstrations and narrow instruction coverage) [59, 34, 12, 19, 35]. This setting exposes a performance-generality dilemma for standard supervised fine-tuning (SFT): it can learn the target skill from limited demonstrations, yet often over-specializes to the post-training data distribution, making the learned skill hard to steer under novel instructions beyond the post-training conditions [47, 14, 60, 54, 26, 58].
To mitigate this failure, existing literature often introduces additional supervision signals, such as those derived from foundation models [34, 38, 11, 32, 27] or auxiliary objectives (e.g., VQA-style losses, dynamics prediction losses), or relies on augmented datasets (e.g., human-robot co-training) to improve generalization [16, 31, 45]. However, these dependencies increase system complexity as well as training and deployment cost [7, 39], and they sidestep a key question: whether a VLA’s internal pre-trained priors can be preserved and effectively leveraged to enable post-training generalization under realistic low-data settings, where exhaustive data coverage over concepts and spatial variants is unavailable. In this work, we formalize this common post-training failure mode in VLAs under the unified concept of lock-in, as shown in Figure 1. Concretely, it manifests in two forms: concept lock-in, where the policy collapses its language grounding and fixates on the training concept(s) (e.g., specific object identities or attributes), executing the learned skill on them regardless of which concept the prompt specifies, and spatial lock-in, where the policy remains anchored to the training spatial target(s) (e.g., left vs. right, upper vs. lower) and continues acting toward those locations regardless of the spatial instruction in the prompt.

Figure 1: Lock-In Failure Mode. In low-data post-training, VLA policies can over-specialize into the training-demo distribution, becoming difficult to steer under novel prompts. We highlight concept lock-in under novel object concepts and spatial lock-in under novel spatial relations. Blue and green arrows denote desired and executed trajectories, respectively.

In this paper, we introduce DeLock, a simple yet effective framework for low-data post-training that preserves a generalist policy’s concept and spatial generalization under narrow demonstrations and instruction coverage. DeLock combines two tightly coupled ingredients: (i) visual encoder weight-drift regularization during post-training to preserve pre-trained visual grounding and prevent representation collapse toward a narrow post-training distribution, and (ii) contrastive prompt guidance (CPG), a classifier-free prompt-contrast rule [41, 2, 24, 52, 25] applied at test time to steer action generation using the model’s own conditional denoising dynamics. Specifically, the regularized model effectively improves steerability to novel concepts (e.g., unseen object identities or attributes), while retaining the grounding signal needed for contrastive prompt guidance. In CPG, the trained prompt defines the negative condition, reflecting post-training bias toward training targets, while the novel prompt defines the positive condition. The difference between their denoising flows is then used as a test-time guidance signal to steer execution toward the novel instruction. To systematically evaluate concept- and spatial-reasoning generalization under low-data post-training, we further develop a dedicated benchmark spanning both simulation and real-world settings.
In summary, our contributions are threefold:
(1) We formalize lock-in as a common low-data post-training failure mode in VLAs, and distinguish two forms: concept lock-in and spatial lock-in. These failure modes manifest as broken language grounding and degraded instruction-following generalization beyond the post-training distribution.
(2) We propose DeLock, a simple yet effective framework for low-data post-training that jointly preserves and exploits a VLA’s internal pre-trained priors: visual encoder weight-drift regularization preserves pre-trained grounding during post-training, while contrastive prompt guidance (CPG) leverages the preserved grounding to steer the policy at test time.
(3) Since no existing benchmark directly probes these lock-in failures, we build a new evaluation suite with four LIBERO-based [37] simulation tasks and four real-world tasks on the DROID setup [28], and show that DeLock consistently outperforms strong baselines, matching or exceeding the performance of a state-of-the-art generalist policy (e.g., π0.5​-DROID\pi_{0.5}\text{-DROID} [23]) post-trained with substantially more curated demonstrations.



2 Related Work


Post-Training Generalist Policies in Low-Data Regime.

Generalist policies can solve diverse tasks, but achieving strong performance on a target skill typically requires post-training on task demonstrations [61, 1, 10]. Prior adaptation paradigms include supervised fine-tuning [6, 50], in-context test-time improvement [15, 46], and RL-based adaptation [43, 55, 22, 20, 33]; however, in the common low-data setting, post-training often causes the policy to over-specialize to dataset-specific biases, making the learned skill difficult to steer under novel instructions [56, 14]. In contrast, DeLock preserves pre-trained visual grounding during low-data post-training and enables contrast prompt guidance for generalization beyond the post-training instructions.


Inference-Time Guidance for Generative Policies.

After post-training, policy behavior can still be shaped at rollout time without modifying weights [53, 9]. Prior work achieves such steering by introducing external guidance sources, such as learned dynamics models that impose predictive constraints [13, 48, 21], value signals that bias sampling [30, 40, 51], or human/VLM feedback that selects or filters candidate behaviors [57, 38, 36]. They introduce additional training/deployment dependencies and can inherit biases from auxiliary objectives. DeLock instead steers via the policy’s own pre-trained grounding, using prompt contrast as an internal guidance signal without auxiliary models or human/VLM intervention.




3 Method


Figure 2: Test-Time Contrastive Prompt Guidance. DeLock uses contrastive prompt guidance at inference time to steer a post-trained flow-based VLA toward novel instructions.
(a) At rollout step kk, the policy conditions on observation oko_{k} and iteratively denoises a noised action chunk akta_{k}^{t} from t=1t{=}1 to 0 (with δ<0\delta<0). At each denoising step, the same policy is forwarded twice with a positive (novel) prompt τ+\tau^{+} and a negative (trained) prompt τ−\tau^{-} to obtain vτ+tv_{\tau^{+}}^{t} and vτ−tv_{\tau^{-}}^{t}, which are combined as vCPGt=vτ−t+w​(vτ+t−vτ−t)v_{\mathrm{CPG}}^{t}=v_{\tau^{-}}^{t}+w\!\left(v_{\tau^{+}}^{t}-v_{\tau^{-}}^{t}\right) to update akt+δ=akt+δ​vCPGta_{k}^{t+\delta}=a_{k}^{t}+\delta\,v_{\mathrm{CPG}}^{t}.
(b) An example of one denoising step: the prompt contrast steers the denoising trajectory away from the trained mode toward a mode aligned with the novel instruction. Detailed pseudocode is provided in Appendix A.



3.1 Problem Formulation

Given a pre-trained generalist policy, our goal is to post-train it on a small, narrowly covered demonstration set for a target skill while avoiding lock-in.

Standard SFT.

A VLA policy πθ​(a∣o,τ)\pi_{\theta}(a\mid o,\tau) maps an observation o∈𝒪o\in\mathcal{O} and instruction τ∈𝒯\tau\in\mathcal{T} to an action distribution over a∈𝒜a\in\mathcal{A}. Pre-training yields πθpre\pi_{\theta_{\mathrm{pre}}} by minimizing ℒpre​(θ;𝒟pre)\mathcal{L}_{\mathrm{pre}}(\theta;\mathcal{D}_{\mathrm{pre}}) on a large-scale dataset 𝒟pre={(oi,ai,τi)}i=1Npre\mathcal{D}_{\mathrm{pre}}=\{(o_{i},a_{i},\tau_{i})\}_{i=1}^{N_{\mathrm{pre}}}, giving θpre\theta_{\mathrm{pre}}. In low-data post-training, we are given 𝒟⋆={(oj,aj,τj)}j=1N⋆\mathcal{D}_{\star}=\{(o_{j},a_{j},\tau_{j})\}_{j=1}^{N_{\star}} with N⋆≪NpreN_{\star}\ll N_{\mathrm{pre}} and narrow instruction coverage τj∈𝒯⋆⊂𝒯\tau_{j}\in\mathcal{T}_{\star}\subset\mathcal{T} (often from a fixed scene with restricted concept/spatial variants). Standard SFT initializes from θpre\theta_{\mathrm{pre}} and minimizes the behavioral-cloning loss ℒBC​(θ;𝒟⋆)=−𝔼(o,a,τ)∼𝒟⋆​[log⁡πθ​(a∣o,τ)]\mathcal{L}_{\mathrm{BC}}(\theta;\mathcal{D}_{\star})=-\mathbb{E}_{(o,a,\tau)\sim\mathcal{D}_{\star}}\!\left[\log\pi_{\theta}(a\mid o,\tau)\right], producing πθft\pi_{\theta_{\mathrm{ft}}}.


Concept Lock-In vs. Spatial Lock-In.

Lock-in manifests when πθft\pi_{\theta_{\mathrm{ft}}} is insensitive to instructions that were not covered in 𝒟⋆\mathcal{D}_{\star}, despite having learned the underlying skill and having seen the underlying concepts in the pretraining data. As illustrated in Figure 1, we categorize this into two types. Concept lock-in occurs when post-training demonstrations involve only a subset of concepts (e.g., only “pick bread” among multiple randomly placed objects); at test time, the policy remains fixated on the training concept even when the prompt specifies a different one (e.g., “pick apple”). Spatial lock-in occurs when demonstrations only cover a specific spatial target (e.g., always “pick right cup” when two cups are present), leading the policy to ignore alternative spatial cues in the prompt (e.g., “pick left cup”). Avoiding such lock-in without exhaustively collecting demonstrations for every concept and spatial variant is the central challenge we address.




3.2 Visual Encoder Weight-Drift Regularization for Knowledge Preservation


Visual Grounding as a Bottleneck for Avoiding Lock-In.

Most modern generalist VLA policies factor into three components: a visual encoder vθvv_{\theta_{v}} that extracts visual features from the observation oo, a language backbone lθℓl_{\theta_{\ell}} that conditions these features on the instruction τ\tau, and an action expert aθaa_{\theta_{a}} that decodes robot actions. We parameterize the policy as a∼πθ(⋅∣o,τ)a\sim\pi_{\theta}(\cdot\mid o,\tau), where the policy is instantiated as aθa​(lθℓ​(τ,vθv​(o)))a_{\theta_{a}}\!\big(l_{\theta_{\ell}}(\tau,\;v_{\theta_{v}}(o))\big). In standard low-data SFT, practitioners often use LoRA to prevent over-updating the language backbone and action expert, but still allow the visual encoder to update freely. Since instruction grounding to scene concepts and spatial relations is mediated by the visual features vθv​(o)v_{\theta_{v}}(o), drift in the visual encoder can directly impair this grounding, and downstream (even general) language/action modules have limited ability to recover it. We therefore seek to preserve the pre-trained visual priors while still allowing task-specific adaptation in the downstream modules.


L2 Regularization on Visual Encoder Drift.


Let θvpre\theta_{v}^{\mathrm{pre}} denote the visual encoder parameters of the pre-trained model and θv\theta_{v} the visual encoder parameters during post-training. DeLock augments the SFT objective with an L2L_{2} penalty that discourages θv\theta_{v} from drifting far from θvpre\theta_{v}^{\mathrm{pre}}:




ℒDeLock​(θ;𝒟⋆)=ℒBC​(θ;𝒟⋆)+λ​∥θv−θvpre∥22,\mathcal{L}_{\texttt{DeLock}}(\theta;\mathcal{D}_{\star})=\mathcal{L}_{\mathrm{BC}}(\theta;\mathcal{D}_{\star})+\lambda\big\lVert\theta_{v}-\theta_{v}^{\mathrm{pre}}\big\rVert_{2}^{2},

(1)


where λ\lambda controls the regularization strength. We regularize the visual encoder parameters θv\theta_{v} during post-training and adapt lθℓl_{\theta_{\ell}} and aθaa_{\theta_{a}} as in standard low-data SFT (e.g., via LoRA), preserving pre-trained visual grounding while acquiring the target skill.





3.3 Contrastive Prompt Guidance at Test Time

Building on the visually preserved model, we introduce contrastive prompt guidance (CPG), a test-time prompt-contrast rule that leverages the policy’s retained grounding knowledge to steer its conditional denoising dynamics and better align execution with novel instructions, as shown in Figure 2.

Flow-Matching Action Generation.

Current state-of-the-art VLA policies commonly parameterize action generation as a conditional denoising process [23, 4, 5]. We define t∈[0,1]t\in[0,1] with t=1t=1 corresponding to pure noise and t=0t=0 to the target action distribution. Let akta_{k}^{t} denote the action chunk at rollout step kk, flow time tt, initialized as ak1∼𝒩​(0,I)a_{k}^{1}\sim\mathcal{N}(0,I). Conditioned on observation oko_{k} and instruction τ\tau, the policy predicts a denoising vector field vθ​(ok,τ,t)v_{\theta}(o_{k},\tau,t), and generates actions by integrating this field via an Euler update rule:
akt+δ=akt+δ​vθ​(ok,τ,t),a_{k}^{t+\delta}=a_{k}^{t}+\delta\,v_{\theta}(o_{k},\tau,t),
where δ<0\delta<0 and we iterate from t=1t=1 to t=0t=0 to obtain the final action ak=ak0a_{k}=a_{k}^{0}.


Prompt-Contrast Steering Rule.


DeLock steers the denoising dynamics by contrasting a positive prompt τ+\tau^{+}, corresponding to the novel instruction, with a negative prompt τ−\tau^{-}. In this work, we use the trained instruction as the negative prompt, as it captures the post-training bias toward the training targets. Leveraging the retained grounding from Section 3.2, we use the prompt-induced contrast between the two conditional vector fields as an inference-time guidance signal. Concretely, we define the guided field as




vCPG,kt=vθ​(ok,τ−,t)+w​(vθ​(ok,τ+,t)−vθ​(ok,τ−,t)),v_{\mathrm{CPG},k}^{t}=v_{\theta}(o_{k},\tau^{-},t)+w\Big(v_{\theta}(o_{k},\tau^{+},t)-v_{\theta}(o_{k},\tau^{-},t)\Big),

(2)


where w≥0w\geq 0 is a tunable guidance scale, and apply it by replacing the Euler update with
akt+δ=akt+δ​vCPG,kt.a_{k}^{t+\delta}=a_{k}^{t}+\delta\,v_{\mathrm{CPG},k}^{t}.
This amplifies the contrastive signal vτ+−vτ−v_{\tau^{+}}-v_{\tau^{-}}, strengthening instruction-relevant directions while suppressing shared bias, so the guidance remains effective even when vτ+v_{\tau^{+}} itself is already overfitting to training targets after low-data post-training.






4 Experiments

In this section, we empirically evaluate DeLock and answer three questions: (1) How can we design evaluation benchmarks that systematically probe the lock-in failure introduced above? (2) What are the representation- and denoising-dynamics signatures of lock-in, and how does DeLock mitigate them? (3) How does DeLock compare with strong low-data post-training baselines and with generalist policies post-trained on substantially larger curated demonstration sets? Unless otherwise stated, all experiments start from a pre-trained flow-based VLA model π0.5​-BASE\pi_{0.5}\text{-BASE} [23] and apply low-data post-training under the protocols described below. Training details are provided in Appendix B.


4.1 Benchmarking Lock-In under Low-Data Post-Training


Figure 3: Lock-In Failure Evaluation Benchmark. Our 8-task suite spans four LIBERO simulation tasks and four real-world DROID tasks. Labels [C] and [S] denote concept- and spatial-lock-in probes, respectively. Yellow arrows illustrate the manipulation pattern demonstrated during post-training. In tasks with shaded regions, we additionally evaluate OOD location shift: the green-shaded region indicates the object placement distribution in post-training demonstrations, and the blue-shaded region indicates the shifted placement distribution used at evaluation.

A key challenge in studying lock-in is that many existing benchmarks primarily test how well a post-trained policy performs the trained task, often using the same instructions as in post-training [37, 34, 10]. While these benchmarks do evaluate generalization, their scope is typically limited to visual or spatial distribution shifts—such as out-of-distribution (OOD) object locations, background variations, or the presence of irrelevant distractors—rather than whether the policy remains responsive to changes in the instruction itself. As a result, strong performance under visual diversity can mask a policy’s inability to re-steer a learned skill when only the instruction changes.
To address this gap, we introduce an 8-task evaluation suite spanning both simulation and the real world, as shown in Figure 3. With one exception that focuses purely on a standard OOD location shift (MokaPot-on-Stove), all other tasks are designed as paired lock-in probes: post-training demonstrations cover a restricted set of concept and/or spatial variants, while evaluation keeps the scene fixed and changes only the concept token or the spatial token in the instruction. The suite includes four LIBERO-based simulation tasks (100 demonstrations per task) and four real-world tasks on the DROID setup (80 demonstrations per task). Table 1 details the specific contrast between post-training and novel evaluation prompts.

Table 1: Instruction-level Shifts for Probing Lock-In Failure. We show the post-training prompts and the novel evaluation prompts for each task. Italicized tokens mark the instruction components varied between post-training and evaluation. For prompts with brackets, one listed variant is sampled per trial. Detailed task designs are provided in Appendix C.





4.2 Mechanistic Analysis: Signatures of Lock-In

We qualitatively analyze lock-in through two complementary lenses: (i) representation-level grounding, and (ii) generation-dynamics bias during action denoising, as illustrated in Figure 4.

Figure 4: Qualitative Evaluation of Lock-In Failure.
(a) Block-Stacking [C], from “stack blue block on green block” to “stack green block on blue block”. Standard SFT shows weak prompt-conditioned attention shift, while DeLock exhibits clearer instruction-aligned attention reallocation.
(b) Cup-to-Box [S], evaluated on the novel prompt “put left cup to box”. The red curve shows the observed rollout with DeLock (CPG enabled), while the green arrows show a counterfactual replay on the same observations without CPG, which remains biased toward the trained target (right cup).


Collapsed Visual Grounding Induces Concept Lock-In.

We use Block-Stacking [C] as a controlled probe for concept lock-in: post-training demonstrations only contain “stack blue block on green block” while evaluation reverses the concept order to “stack green block on blue block”. Although the required stacking skill is unchanged, standard SFT repeats the training behavior, whereas DeLock follows the new instruction and succeeds. To examine the representation-level cause, we visualize vision-language cross-attention in the PaliGemma [3] backbone using instruction tokens as queries and image patches as keys (Figure 4(a)). Standard SFT shows a collapsed attention pattern, continuing to focus on the blue block regardless of the prompt. In contrast, DeLock exhibits a clear prompt-conditioned shift in attention between the blue and green blocks as their instructed roles are swapped, consistent with preserved visual grounding under low-data post-training.


Spatial Lock-In as Biased Denoising Dynamics.

We analyze spatial lock-in on Cup-to-Box [S], where post-training demonstrations only cover picking the right cup and moving it to the box. At evaluation, the instruction is changed to “put left cup to box”. Standard SFT remains biased toward the trained rightward behavior, whereas DeLock successfully reuses the learned pick-and-place skill and redirects it to the left cup. To understand the role of CPG, we perform a counterfactual rollout analysis (Figure 4(b)). Using the same observation sequence, removing CPG causes the policy to continue producing actions biased toward the trained target (green arrows). In contrast, CPG adds a prompt-contrast term that strengthens the instruction-change direction and steers the denoising trajectory toward the new spatial target.




4.3 Generalizing Beyond Post-Training: OOD Configurations and Novel Instructions

We quantitatively evaluate OOD performance, comparing DeLock against (i) a strong low-data post-training baseline, (ii) a large-scale post-trained generalist reference, and (iii) targeted ablations to isolate the contributions of visual encoder regularization and contrastive prompt guidance. Please refer to Appendix D for more experimental results.

Baselines.

(1) RETAIN, a strong low-data post-training method that employs weight-space interpolation between the task-specific and pre-trained models to mitigate catastrophic forgetting; and (2) π0.5​-DROID\pi_{0.5}\text{-DROID}, a state-of-the-art generalist VLA post-trained on the large-scale, curated DROID dataset, which serves as a high-resource reference rather than a low-data baseline, representing the upper bound achieved through massive data curation.


Ablations.

(1) DeLock w/o CPG, which removes the contrastive guidance at inference to assess the impact of raw policy grounding; (2) DeLock w/o Vis-Reg, which fine-tunes the visual encoder without regularization to evaluate the necessity of grounding preservation; and (3) DeLock w/ Frozen-Vis, which fixes the visual encoder during training as a rigid alternative to our regularized adaptation.

Table 2: Generalization under OOD Locations and Novel Instructions (20 trials per task). We report success counts for spatial generalization under OOD location shifts (T1, T4) and for generalization to novel prompts across the seven lock-in probes. Task IDs follow Table 1; π0.5​-DROID\pi_{0.5}\text{-DROID} is evaluated exclusively in real-world; full in-distribution results are provided in Appendix D.




Method
OOD Locations
OOD Instructions (Novel Prompts)

T1
T4 [C]
T2 [C]
T3 [C]
T4 [C]
T5 [S]
T6 [S]
T7 [S]
T8 [C+S]

RETAIN
10/20
14/20
0/20
6/20
3/20
0/20
0/20
2/20
1/20

π0.5​-DROID\pi_{0.5}\text{-DROID}
–
18/20
–
18/20
18/20
–
–
11/20
0/20

DeLock w/o CPG
16/20
16/20
17/20
18/20
15/20
0/20
0/20
0/20
0/20

DeLock w/o Vis-Reg
4/20
9/20
9/20
7/20
2/20
0/20
0/20
0/20
0/20

DeLock w/ Frozen-Vis
7/20
13/20
16/20
14/20
13/20
2/20
11/20
8/20
4/20

DeLock
16/20
15/20
19/20
19/20
17/20
11/20
13/20
14/20
13/20



Table 2 summarizes the success rates under both OOD spatial configurations and novel-instruction generalization. Overall, DeLock demonstrates superior across-the-board performance: it maintains high reliability under scene perturbations and significantly outperforms baselines on both concept- and spatial-lock-in probes. Figure 5 qualitatively compares DeLock and RETAIN in their ability to follow novel prompts across two challenging articulated tasks.

Figure 5: Novel-Prompt Rollouts on Articulated Tasks. We compare DeLock and RETAIN on the two challenging tasks involving articulated objects: Open-Microwave [S] and Open-Labeled-Door [C+S]. Under novel prompts, RETAIN largely repeats the post-training trajectory and fails to follow the changed spatial/concept specification, whereas DeLock successfully re-steers the learned skill to follow the new instruction.

Among the baselines, RETAIN exhibits moderate robustness to OOD location shifts, suggesting that parameter merging can partially mitigate sensitivity to visual-spatial distribution shifts. However, it fails to generalize across all novel-prompt tasks, indicating that simple weight-space interpolation is insufficient to overcome behavioral lock-in under instruction changes. In contrast, while π0.5​-DROID\pi_{0.5}\text{-DROID} performs well on OOD locations and some simpler concept shifts, its performance depends on an extensive post-training distribution that is often unavailable in specialized domains. Moreover, its performance drops substantially on spatial reasoning tasks such as Cup-to-Box [S], where its behavior resembles stochastic switching between targets, and it fails entirely on the more fine-grained Open-Labeled-Door [C+S] task. Together, these results suggest that even large-scale generalists struggle when adaptation requires fine-grained instruction following beyond dominant post-training patterns, highlighting the necessity for effective task-specific adaptation in the low-data regime.
Our ablation studies further disentangle the contributions of each component. Removing visual encoder regularization (DeLock w/o Vis-Reg) substantially degrades performance under both OOD location shifts and novel-prompt settings, confirming that preserving pre-trained grounding is critical for robust downstream generalization. However, preservation alone is not sufficient: removing CPG (DeLock w/o CPG) retains some success on concept probes but fails on all spatial-lock-in tasks, showing that prompt-contrast steering is essential for correcting training-induced spatial bias during rollout. Finally, fully freezing the visual encoder (DeLock w/ Frozen-Vis) yields non-trivial performance but consistently underperforms DeLock, underscoring the benefit of controlled, regularized adaptation over rigid parameter freezing. Together, these results show that DeLock’s effectiveness arises from the complementary roles of grounding preservation and test-time steering.





5 Conclusion

In this paper, we formalize lock-in as a common failure mode of low-data post-training in generalist VLA policies, and distinguish two forms: concept lock-in and spatial lock-in. We show that, under limited post-training data, policies can over-specialize to demonstration biases and lose the ability to re-steer learned skills under novel instructions. To address this, we introduce DeLock, which combines visual encoder weight-drift regularization to preserve pre-trained grounding with test-time contrastive prompt guidance to steer execution. Across both simulated and real-world tasks, our results show that this preserve-and-steer design enables effective instruction-conditioned OOD generalization without relying on large-scale post-training data or external sources of supervision.

Limitations.

Our study focuses on lock-in under relatively controlled low-data post-training settings, where the trained and novel prompts are specified a priori for contrastive guidance, and it remains to be seen how the same mechanisms scale to broader instruction distributions, longer-horizon tasks, and more open-ended real-world environments. In addition, our current visual encoder regularization and guidance scale design are relatively simple; more adaptive regularization schemes and context-dependent guidance strategies may further improve the trade-off between grounding preservation and task-specific adaptation.




References



[1]
R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. (2023)

Palm 2 technical report.

arXiv preprint arXiv:2305.10403.

Cited by: §2.



[2]
 (2024)

Understanding the impact of negative prompts: when and how do they take effect?.

In european conference on computer vision,

 pp. 190–206.

Cited by: §1.



[3]
L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, et al. (2024)

Paligemma: a versatile 3b vlm for transfer.

arXiv preprint arXiv:2407.07726.

Cited by: Appendix B,
§4.2.



[4]
J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)

Gr00t n1: an open foundation model for generalist humanoid robots.

arXiv preprint arXiv:2503.14734.

Cited by: §1,
§3.3.



[5]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)

Pi_0: a vision-language-action flow model for general robot control.

arXiv preprint arXiv:2410.24164.

Cited by: §1,
§3.3.



[6]
K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine (2023)

Zero-shot robotic manipulation with pretrained image-editing diffusion models.

arXiv preprint arXiv:2310.10639.

Cited by: §2.



[7]
R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. (2021)

On the opportunities and risks of foundation models.

arXiv preprint arXiv:2108.07258.

Cited by: §1.



[8]
K. Bousmalis, G. Vezzani, D. Rao, C. Devin, A. X. Lee, M. Bauzá, T. Davchev, Y. Zhou, A. Gupta, A. Raju, et al. (2023)

Robocat: a self-improving generalist agent for robotic manipulation.

arXiv preprint arXiv:2306.11706.

Cited by: §1.



[9]
J. Cao, Y. Huang, H. Guo, R. Zhang, M. Nan, W. Mai, J. Wang, H. Cheng, J. Sun, G. Han, et al. (2025)

Compose your policies! improving diffusion-based or flow-based robot policies via test-time distribution-level composition.

arXiv preprint arXiv:2510.01068.

Cited by: §2.



[10]
Q. Chen, J. Yu, M. Schwager, P. Abbeel, Y. Shentu, and P. Wu (2025)

SARM: stage-aware reward modeling for long horizon robot manipulation.

arXiv preprint arXiv:2509.25358.

Cited by: §2,
§4.1.



[11]
W. Chen, J. S. Bhatia, C. Glossop, N. Mathihalli, R. Doshi, A. Tang, D. Driess, K. Pertsch, and S. Levine (2026)

Steerable vision-language-action policies for embodied reasoning and hierarchical control.

arXiv preprint arXiv:2602.13193.

Cited by: §1.



[12]
B. Cheng, T. Liang, S. Huang, M. Shao, F. Zhang, B. Xu, Z. Xue, and H. Xu (2025)

MoE-dp: an moe-enhanced diffusion policy for robust long-horizon robotic manipulation with skill decomposition and failure recovery.

arXiv preprint arXiv:2511.05007.

Cited by: §1.



[13]
M. Du and S. Song (2025)

Dynaguide: steering diffusion polices with active dynamic guidance.

arXiv preprint arXiv:2506.13922.

Cited by: §2.



[14]
S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, et al. (2025)

Libero-plus: in-depth robustness analysis of vision-language-action models.

arXiv preprint arXiv:2510.13626.

Cited by: §1,
§2.



[15]
L. Fu, H. Huang, G. Datta, L. Y. Chen, W. C. Panitch, F. Liu, H. Li, and K. Goldberg (2024)

In-context imitation learning via next-token prediction.

arXiv preprint arXiv:2408.15980.

Cited by: §2.



[16]
S. Grover, A. Gopalkrishnan, B. Ai, H. I. Christensen, H. Su, and X. Li (2025)

Enhancing generalization in vision-language-action models by preserving pretrained representations.

arXiv preprint arXiv:2509.11417.

Cited by: §1.



[17]
Y. Guo, T. Lee, L. X. Shi, J. Chen, P. Liang, and C. Finn (2026)

VLAW: iterative co-improvement of vision-language-action policy and world model.

arXiv preprint arXiv:2602.12063.

Cited by: §1.



[18]
Y. Guo, L. X. Shi, J. Chen, and C. Finn (2025)

Ctrl-world: a controllable generative world model for robot manipulation.

arXiv preprint arXiv:2510.10125.

Cited by: §1.



[19]
Y. Guo, J. Zhang, X. Chen, X. Ji, Y. Wang, Y. Hu, and J. Chen (2025)

Improving vision-language-action model with online reinforcement learning.

In 2025 IEEE International Conference on Robotics and Automation (ICRA),

 pp. 15665–15672.

Cited by: §1.



[20]
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. (2022)

Lora: low-rank adaptation of large language models..

Iclr 1 (2),  pp. 3.

Cited by: Appendix B,
§2.



[21]
S. Huang, Q. Chen, X. Zhang, J. Sun, and M. Schwager (2025)

Particleformer: a 3d point cloud world model for multi-object, multi-material robotic manipulation.

arXiv preprint arXiv:2506.23126.

Cited by: §2.



[22]
S. Huang, Z. Zhang, T. Liang, Y. Xu, Z. Kou, C. Lu, G. Xu, Z. Xue, and H. Xu (2024)

Mentor: mixture-of-experts network with task-oriented perturbation for visual reinforcement learning.

arXiv preprint arXiv:2410.14972.

Cited by: §2.



[23]
P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025)

Pi_05: a vision-language-action model with open-world generalization.

pi05: a vision-language-action model with open-world generalization.

Cited by: Appendix B,
§1,
§1,
§3.3,
§4.



[24]
J. Jang, S. Ye, and M. Seo (2023)

Can large language models truly understand prompts? a case study with negated prompts.

In Transfer learning for natural language processing workshop,

 pp. 52–62.

Cited by: §1.



[25]
J. Jeong, J. Kim, G. Lee, Y. Choi, and Y. Uh (2025)

StyleKeeper: prevent content leakage using negative visual query guidance.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

 pp. 15760–15769.

Cited by: §1.



[26]
X. Jin, X. Ren, D. Preotiuc-Pietro, and P. Cheng (2022)

Dataless knowledge fusion by merging weights of language models.

arXiv preprint arXiv:2212.09849.

Cited by: §1.



[27]
N. Kachaev, M. Kolosov, D. Zelezetsky, A. K. Kovalev, and A. I. Panov

Don’t blind your vla: aligning visual representations for ood generalization, 2025.

URL https://arxiv. org/abs/2510.25616 2 (4).

Cited by: §1.



[28]
A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. (2024)

Droid: a large-scale in-the-wild robot manipulation dataset.

arXiv preprint arXiv:2403.12945.

Cited by: §C.1,
§1,
§1.



[29]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)

Openvla: an open-source vision-language-action model.

arXiv preprint arXiv:2406.09246.

Cited by: §1.



[30]
F. Koulischer, J. Deleu, G. Raya, T. Demeester, and L. Ambrogioni (2024)

Dynamic negative guidance of diffusion models.

arXiv preprint arXiv:2410.14398.

Cited by: §2.



[31]
M. Lepert, J. Fang, and J. Bohg (2025)

Masquerade: learning from in-the-wild human videos using data-editing.

arXiv preprint arXiv:2508.09976.

Cited by: §1.



[32]
F. Li, W. Song, H. Zhao, J. Wang, P. Ding, D. Wang, L. Zeng, and H. Li (2025)

Spatial forcing: implicit spatial representation alignment for vision-language-action model.

arXiv preprint arXiv:2510.12276.

Cited by: §D.4,
§1.



[33]
H. Li, Y. Zuo, J. Yu, Y. Zhang, Z. Yang, K. Zhang, X. Zhu, Y. Zhang, T. Chen, G. Cui, et al. (2025)

Simplevla-rl: scaling vla training via reinforcement learning.

arXiv preprint arXiv:2509.09674.

Cited by: §2.



[34]
P. Li, Y. Wu, Z. Xi, W. Li, Y. Huang, Z. Zhang, Y. Chen, J. Wang, S. Zhu, T. Liu, et al. (2025)

Controlvla: few-shot object-centric adaptation for pre-trained vision-language-action models.

arXiv preprint arXiv:2506.16211.

Cited by: §1,
§1,
§4.1.



[35]
X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. (2024)

Evaluating real-world robot manipulation policies in simulation.

arXiv preprint arXiv:2405.05941.

Cited by: §1.



[36]
Z. Li, J. Liu, Z. Dong, T. Teng, Q. Rouxel, D. Caldwell, and F. Chen (2025)

Towards deploying vla without fine-tuning: plug-and-play inference-time vla policy steering via embodied evolutionary diffusion.

arXiv preprint arXiv:2511.14178.

Cited by: §2.



[37]
B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone (2023)

Libero: benchmarking knowledge transfer for lifelong robot learning.

Advances in Neural Information Processing Systems 36,  pp. 44776–44791.

Cited by: §C.1,
§1,
§4.1.



[38]
S. Liu, I. S. Singh, Y. Xu, J. Duan, and R. Krishna (2026)

VLS: steering pretrained robot policies via vision-language models.

arXiv preprint arXiv:2602.03973.

Cited by: §1,
§2.



[39]
Y. Ma, Z. Song, Y. Zhuang, J. Hao, and I. King (2024)

A survey on vision-language-action models for embodied ai.

arXiv preprint arXiv:2405.14093.

Cited by: §1.



[40]
M. Nakamoto, O. Mees, A. Kumar, and S. Levine (2024)

Steering your generalists: improving robotic foundation models via value guidance.

arXiv preprint arXiv:2410.13816.

Cited by: §2.



[41]
T. Nguyen, M. N. Vu, B. Huang, A. Vuong, Q. Vuong, N. Le, T. Vo, and A. Nguyen (2024)

Language-driven 6-dof grasp detection using negative prompt guidance.

In European Conference on Computer Vision,

 pp. 363–381.

Cited by: §1.



[42]
A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. (2024)

Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0.

In 2024 IEEE International Conference on Robotics and Automation (ICRA),

 pp. 6892–6903.

Cited by: §1.



[43]
M. Pan, S. Feng, Q. Zhang, X. Li, J. Song, C. Qu, Y. Wang, C. Li, Z. Xiong, Z. Chen, et al. (2026)

SOP: a scalable online post-training system for vision-language-action models.

arXiv preprint arXiv:2601.03044.

Cited by: §2.



[44]
K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025)

Fast: efficient action tokenization for vision-language-action models.

arXiv preprint arXiv:2501.09747.

Cited by: §1.



[45]
R. Punamiya, D. Patel, P. Aphiwetsa, P. Kuppili, L. Y. Zhu, S. Kareer, J. Hoffman, and D. Xu (2025)

Egobridge: domain adaptation for generalizable imitation from egocentric human data.

In Human to Robot: Workshop on Sensorizing, Modeling, and Learning from Humans,

Cited by: §1.



[46]
M. Sharma, C. Fantacci, Y. Zhou, S. Koppula, N. Heess, J. Scholz, and Y. Aytar (2023)

Lossless adaptation of pretrained vision models for robotic manipulation.

arXiv preprint arXiv:2304.06600.

Cited by: §2.



[47]
I. Shenfeld, J. Pari, and P. Agrawal (2025)

Rl’s razor: why online reinforcement learning forgets less.

arXiv preprint arXiv:2509.04259.

Cited by: §1.



[48]
Z. Sun and S. Song (2025)

Latent policy barrier: learning robust visuomotor policies by staying in-distribution.

arXiv preprint arXiv:2508.05941.

Cited by: §2.



[49]
G. R. Team, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. (2025)

Gemini robotics: bringing ai into the physical world.

arXiv preprint arXiv:2503.20020.

Cited by: §1.



[50]
O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. (2024)

Octo: an open-source generalist robot policy.

arXiv preprint arXiv:2405.12213.

Cited by: §2.



[51]
A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine (2025)

Steering your diffusion policy with latent space reinforcement learning.

arXiv preprint arXiv:2506.15799.

Cited by: §2.



[52]
D. Wan, J. Cho, E. Stengel-Eskin, and M. Bansal (2024)

Contrastive region guidance: improving grounding in vision-language models without training.

In European Conference on Computer Vision,

 pp. 198–215.

Cited by: §1.



[53]
L. Wang, J. Zhao, Y. Du, E. H. Adelson, and R. Tedrake (2024)

Poco: policy composition from and for heterogeneous robot learning.

arXiv preprint arXiv:2402.02511.

Cited by: §2.



[54]
M. Wortsman, G. Ilharco, J. W. Kim, M. Li, S. Kornblith, R. Roelofs, R. G. Lopes, H. Hajishirzi, A. Farhadi, H. Namkoong, et al. (2022)

Robust fine-tuning of zero-shot models.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

 pp. 7959–7971.

Cited by: §1.



[55]
T. Xiang, A. Jin, X. Zhou, M. Gui, X. Xie, S. Liu, S. Wang, S. Duan, F. Xie, W. Wang, et al. (2025)

Parallels between vla model post-training and human motor learning: progress, challenges, and trends.

arXiv preprint arXiv:2506.20966.

Cited by: §2.



[56]
Y. Xing, X. Luo, J. Xie, L. Gao, H. Shen, and J. Song (2025)

Shortcut learning in generalist robot policies: the role of dataset diversity and fragmentation.

arXiv preprint arXiv:2508.06426.

Cited by: §2.



[57]
M. Xu, Z. Xu, C. Chi, M. Veloso, and S. Song (2023)

Xskill: cross embodiment skill discovery.

In Conference on robot learning,

 pp. 3536–3555.

Cited by: §2.



[58]
Y. Yadav, Z. Zhou, A. Wagenmaker, K. Pertsch, and S. Levine (2025)

Robust finetuning of vision-language-action robot policies via parameter merging.

arXiv preprint arXiv:2512.08333.

Cited by: §1.



[59]
H. Zang, M. Wei, S. Xu, Y. Wu, Z. Guo, Y. Wang, H. Lin, L. Shi, Y. Xie, Z. Xu, et al. (2025)

Rlinf-vla: a unified and efficient framework for vla+ rl training.

arXiv preprint arXiv:2510.06710.

Cited by: §1.



[60]
X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun (2025)

LIBERO-pro: towards robust and fair evaluation of vision-language-action models beyond memorization.

arXiv preprint arXiv:2510.03827.

Cited by: §1.



[61]
B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)

Rt-2: vision-language-action models transfer web knowledge to robotic control.

In Conference on Robot Learning,

 pp. 2165–2183.

Cited by: §2.






Appendix




Appendix A Pseudocode for Training and Inference

For completeness, we provide pseudocode for the two core components of DeLock. Algorithm 1 describes the training-time procedure, where the visual encoder is regularized toward its pre-trained parameters while the language backbone and action expert are adapted via LoRA. Algorithm 2 describes the test-time contrastive prompt guidance rule used for action denoising under novel instructions.

Algorithm 1  Low-Data Post-Training with Visual Encoder Weight-Drift Regularization


1:pre-trained policy parameters θvpre,θℓpre,θapre\theta_{v}^{\mathrm{pre}},\theta_{\ell}^{\mathrm{pre}},\theta_{a}^{\mathrm{pre}}, low-data post-training set 𝒟⋆\mathcal{D}_{\star}, regularization weight λ\lambda



2:post-trained parameters θv,θℓ,θa\theta_{v},\theta_{\ell},\theta_{a}



3:Initialize visual encoder parameters θv←θvpre\theta_{v}\leftarrow\theta_{v}^{\mathrm{pre}}


4:Initialize language backbone and action expert from pre-trained weights



5:Insert LoRA adapters into lθℓl_{\theta_{\ell}} and aθaa_{\theta_{a}}



6:Freeze base parameters of lθℓl_{\theta_{\ell}} and aθaa_{\theta_{a}} except LoRA parameters



7:Keep a frozen copy of pre-trained visual parameters θvpre\theta_{v}^{\mathrm{pre}} as reference


8:while not converged do



9:  Sample minibatch (o,a,τ)∼𝒟⋆(o,a,\tau)\sim\mathcal{D}_{\star}


10:  Compute policy output:



11:a^∼πθ(⋅∣o,τ)=aθa(lθℓ(τ,vθv(o)))\hat{a}\sim\pi_{\theta}(\cdot\mid o,\tau)=a_{\theta_{a}}\!\big(l_{\theta_{\ell}}(\tau,v_{\theta_{v}}(o))\big)


12:  Compute behavioral cloning loss:



13:ℒBC←−𝔼(o,a,τ)∼𝒟⋆​[log⁡πθ​(a∣o,τ)]\mathcal{L}_{\mathrm{BC}}\leftarrow-\mathbb{E}_{(o,a,\tau)\sim\mathcal{D}_{\star}}\!\left[\log\pi_{\theta}(a\mid o,\tau)\right]


14:  Compute visual encoder drift penalty:



15:ℒreg←λ​∥θv−θvpre∥22\mathcal{L}_{\mathrm{reg}}\leftarrow\lambda\lVert\theta_{v}-\theta_{v}^{\mathrm{pre}}\rVert_{2}^{2}


16:  Form total loss:



17:ℒDeLock←ℒBC+ℒreg\mathcal{L}_{\texttt{DeLock}}\leftarrow\mathcal{L}_{\mathrm{BC}}+\mathcal{L}_{\mathrm{reg}}



18:  Update θv\theta_{v} using ∇θvℒDeLock\nabla_{\theta_{v}}\mathcal{L}_{\texttt{DeLock}}



19:  Update only the LoRA parameters in lθℓl_{\theta_{\ell}} and aθaa_{\theta_{a}}


20:end while


21:returnθv,θℓ,θa\theta_{v},\theta_{\ell},\theta_{a}




We next describe the inference-time steering rule applied to the post-trained policy.

Algorithm 2  Test-Time Contrastive Prompt Guidance for Flow-Based VLA


1:observation oko_{k}, novel prompt τ+\tau^{+}, trained prompt τ−\tau^{-}, guidance scale ww, step size δ<0\delta<0



2:final action chunk ak0a_{k}^{0}



3:Sample initial noisy action chunk ak1∼𝒩​(0,I)a_{k}^{1}\sim\mathcal{N}(0,I)



4:Set denoising time t←1t\leftarrow 1



5:whilet>0t>0 do


6:  Forward the same post-trained policy with the novel prompt:



7:vτ+t←vθ​(ok,τ+,t)v_{\tau^{+}}^{t}\leftarrow v_{\theta}(o_{k},\tau^{+},t)


8:  Forward the same post-trained policy with the trained prompt:



9:vτ−t←vθ​(ok,τ−,t)v_{\tau^{-}}^{t}\leftarrow v_{\theta}(o_{k},\tau^{-},t)


10:  Construct the guided vector field:



11:vCPG,kt←vτ−t+w​(vτ+t−vτ−t)v_{\mathrm{CPG},k}^{t}\leftarrow v_{\tau^{-}}^{t}+w\!\left(v_{\tau^{+}}^{t}-v_{\tau^{-}}^{t}\right)


12:  Update the action chunk:



13:akt+δ←akt+δ​vCPG,kta_{k}^{t+\delta}\leftarrow a_{k}^{t}+\delta\,v_{\mathrm{CPG},k}^{t}



14:t←t+δt\leftarrow t+\delta


15:end while


16:returnak0a_{k}^{0}







Appendix B Post-Training Settings

All VLA post-training experiments in this work are built on the official OpenPI implementation. Unless otherwise specified, all methods are initialized from the same pre-trained model, π0.5​-BASE\pi_{0.5}\text{-BASE} [23], to ensure a controlled comparison across post-training strategies.
For DeLock and all of its self-ablations, we use LoRA-based fine-tuning [20]. Specifically, the PaliGemma backbone [3] is frozen, and the inserted LoRA adapters are updated during post-training. Note, however, that following the official OpenPI implementation, the visual encoder is not frozen and remains fully trainable. In contrast, for RETAIN and π0.5​-DROID\pi_{0.5}\text{-DROID}, we perform full-parameter fine-tuning. The corresponding model variants and LoRA hyperparameters are summarized in Table 3.

Table 3: LoRA Model Variants Used During Post-Training.


In addition, both LoRA-configured modules use a single key-value head (num_kv_heads=1) and a head dimension of 256. LoRA adapters are inserted into both the attention layers and feed-forward layers. Specifically, for the Gemma_2b (VLM part) we use
lora_configs={"attn": rank 16, alpha 16; "ffn": rank 16, alpha 16},
and for the Gemma_300m (action expert part) we use
lora_configs={"attn": rank 32, alpha 32; "ffn": rank 32, alpha 32}.
The remaining post-training hyperparameters are shared across experiments unless otherwise noted. We set the action horizon to 10. Training is performed with batch size 32 for a total of 10,000 optimization steps for each task. We use AdamW with gradient clipping at a global norm of 1.0. Exponential moving average is disabled.
The learning rate follows a cosine decay schedule with 1,000 warmup steps, peak learning rate 5×10−55\times 10^{-5}, and decay steps set to 50,000. Since the decay target is also 5×10−55\times 10^{-5}, the schedule effectively maintains a constant learning rate after warmup over our 10,000-step training horizon. Table 4 summarizes the shared post-training hyperparameters.

Table 4: Shared Post-Training Hyperparameters Used Across Methods.





Appendix C Experimental Implementation Details

Our experiments focus on evaluating the generalization capability of post-trained VLA policies. In particular, we study robotic manipulation scenarios where DeLock is post-trained with only a very small amount of task-specific data (typically around 100 demonstrations for a single task), yet the resulting policy is expected to steer the learned skill toward novel instructions at inference time.
We evaluate our approach across a diverse set of simulated and real-world manipulation tasks. These include pick-and-place and articulated object interaction, requiring the policy to generalize across diverse object categories and complex spatial targets. These tasks are intentionally constructed so that the post-training data covers only a narrow portion of the instruction and environment configuration space, allowing us to evaluate whether the policy can generalize to novel instructions and configurations.
All post-training experiments are conducted on NVIDIA A100 GPUs (80GB). For real-world rollouts, the trained policy is deployed and executed on a workstation equipped with an NVIDIA A5000 GPU.


C.1 Setup


Simulation.

For the simulation experiments, we adopt four environments from the LIBERO benchmark [37]. Because LIBERO was not originally designed to study lock-in effects in low-data VLA post-training, we modify these environments to better expose the failure modes of standard supervised fine-tuning. These adaptations make lock-in behavior more evident, thereby enabling a clearer analysis of the problem setting and a more informative evaluation of our approach.


Real-World.

For the real-world experiments, we follow the DROID hardware setup [28]. The robot platform consists of an Franka Research 3 (FR3) arm with a Robotiq 2F-85 gripper. We use two cameras: a third-person ZED-2i camera that captures the robot and the overall scene geometry, and a ZED Mini mounted on the end-effector flange to provide close-up observations for fine-grained manipulation. As in simulation, the real-world tasks are also designed to expose lock-in failure modes in low-data post-training. The real-world setup is shown in Figure 6.

Figure 6: Real-World Experimental Setup.





C.2 Task Design

The benchmark suite is shown in Figure 3, with detailed design choice as follows:

MokaPot-on-Stove

This simulation task is designed to evaluate the OOD location-shift generalization of post-trained VLA policies. In the training demonstrations, the moka pot is randomly initialized within the green-shaded region, and the robot arm reaches for the pot, grasps it by the handle, and places it onto the stove with language instruction “put moka pot on stove”. The task is deemed successful if the center of mass of the moka pot lies within a predefined threshold of the stove center. At test time, in addition to evaluating the original demonstration setting, we also initialize the moka pot randomly in the blue-shaded region, which is completely out of distribution with respect to object location, and evaluate whether the post-trained policy can still complete the task successfully.


Mug-on-Plate [C]

This simulation task is designed to evaluate the concept lock-in behavior of post-trained VLA policies. In the training demonstrations, multiple mugs are randomly placed on the table, and the robot is instructed to pick and place a specific-colored mug onto a plate. During post-training, the target object is always the green mug, and the training prompt is fixed to “put green mug on plate”. The robot reaches the target mug, grasps it, and places it onto the plate. The task is considered successful when the center of mass of the mug lies within a predefined threshold of the plate center. At test time, in addition to evaluating the original demonstration setting, we replace the target instruction with novel color concepts that are never used as targets during post-training (e.g., “put red mug on plate” or “put blue mug on plate”) and evaluate whether the post-trained policy can follow these instructions correctly. Because the green mug is observed at diverse table locations during training, this task largely factors out location generalization and isolates failure caused by overfitting to the seen object concept.


Block-Stacking [C]

This real-world task is also designed to evaluate the concept lock-in behavior of post-trained VLA policies. Similar to Mug-on-Plate [C], the goal is to test whether a policy post-trained on a single seen concept can still follow novel object-specific instructions at test time. In the training demonstrations, multiple colored blocks are randomly placed on the table, and the robot is instructed to pick one specific block and stack it on top of another fixed target block. During post-training, the prompt is fixed to a single seen object concept, e.g., “stack blue block on green block”. The robot reaches the target block, grasps it, and places it onto the target support block. The task is considered successful when the manipulated block is stably positioned on top of the target block within a predefined spatial threshold. At test time, in addition to evaluating the original demonstration setting, we also prompt the policy with unseen object instructions that refer to different source blocks. This tests whether the post-trained policy can correctly follow novel concept-level instructions rather than overfitting to the seen object identity used during training.


Food-on-Plate [C]

This real-world task likewise evaluates concept lock-in, with the same motivation as in Mug-on-Plate [C]. In the training demonstrations, multiple food objects are randomly placed on the table, and the robot is instructed to pick one specific food item and place it onto a plate. During post-training, the target object is always a single seen concept, with the prompt fixed to “put bread on plate”. The robot reaches the bread, grasps it, and places it onto the plate. The task is considered successful when the center of mass of the food object lies within a predefined threshold of the plate center. At test time, besides evaluating the original demonstration setting, we also replace the target instruction with novel food concepts that are never used as targets during post-training, such as “put apple on plate” or “put carrot on plate”. As in Mug-on-Plate [C], the seen training object appears at diverse locations across demonstrations, so failure in this task primarily reflects overfitting to the seen concept rather than an inability to reach different table locations. In addition, we also evaluate OOD location-shift generalization in this task. During training, the plate is randomly initialized within the green-shaded region, whereas at test time we additionally initialize it within the blue-shaded region.


Open-Microwave [S]

This simulation task is designed to evaluate the spatial lock-in behavior of post-trained VLA policies. Unlike the concept-generalization tasks above, the novel test instructions here do not introduce new object concepts; instead, they require the policy to respond to spatial information expressed in the prompt. The environment contains two microwaves arranged vertically, referred to as the lower microwave and the upper microwave. In the training demonstrations, the robot is instructed to open only the lower microwave, with the prompt fixed to “open lower microwave”. The robot reaches the handle of the lower microwave and executes the corresponding opening motion. The task is considered successful when the target microwave door is opened beyond a predefined angle threshold. At test time, in addition to evaluating the original demonstration setting, we also evaluate the policy under the novel spatial instruction “open upper microwave”. Since the instruction introduces no new object concept and only changes the spatial referent, this task specifically tests whether the post-trained policy can overcome spatial lock-in and generalize the learned manipulation behavior to a previously uncovered region of the workspace. The observations in this task are provided using only wrist-view image.


Mug-on-Plate [S]

This simulation task is designed to evaluate the spatial lock-in behavior of post-trained VLA policies. The object concept remains unchanged throughout training and testing; the challenge is instead whether the policy can follow a novel spatial instruction and move the object to a target region not covered during post-training. In the training demonstrations, the robot is always instructed to place the mug onto one specific plate, with the prompt fixed to “put mug on left plate”. The robot reaches the mug, grasps it, and places it onto the designated plate. The task is considered successful when the center of mass of the mug lies within a predefined threshold of the target plate center. At test time, in addition to evaluating the original demonstration setting, we also evaluate a novel spatial instruction, “put mug on right plate”. Since the instruction introduces no new concept and only changes the spatial target, this task specifically tests whether the post-trained policy can overcome spatial lock-in and generalize the learned manipulation skill to a previously uncovered target region. The observations in this task are provided using only wrist-view image. The left/right concept is defined in wrist-view camera.


Cup-to-Box [S]

This real-world task also evaluates spatial lock-in, following the same rationale as in Mug-on-Plate [S]. In the training demonstrations, the robot is instructed to move a specific cup into a box, with the prompt fixed to “put right cup to box”. The robot reaches the designated cup, grasps it, and places it into the box. The task is considered successful when the targeted cup is placed inside the box. At test time, besides evaluating the original demonstration setting, we also prompt the policy with a novel spatial instruction, “put left cup to box”, which requires the policy to manipulate the same object concept but in a target region not covered during training. This task therefore isolates whether the post-trained policy can use prompt-level spatial information to guide manipulation beyond the narrow spatial support of the demonstrations. The left/right concept is defined in third-view camera.


Open-Labeled-Door [C+S]

This real-world task is designed to evaluate the combined concept and spatial lock-in behavior of post-trained VLA policies. The environment contains a cabinet with two doors that have a fixed spatial relationship: one door is on the left and the other is on the right. To distinguish the two doors, we use visual labels attached to them, e.g., banana and apple images. In the training demonstrations, the robot is instructed to open only one labeled door, with the prompt fixed to a single seen instruction such as “open door labeled with banana”. The robot reaches the corresponding handle and pulls the door open. The task is considered successful when the specified door is opened beyond a predefined threshold. At test time, in addition to evaluating the original demonstration setting, we also prompt the policy to open the other labeled door, e.g., “open door labeled with apple”. This requires the policy to overcome spatial lock-in, since the unseen target door occupies a different spatial region that is never opened during training, and also to overcome concept lock-in, since the two doors are referred to through different visual labels and the novel instruction changes which label should be grounded to action.





Appendix D Additional Experimental Results



D.1 In-Distribution Evaluation Results

Table 5 reports the in-distribution performance of DeLock and all baselines under the trained prompts. The results show that all methods perform well in-distribution after low-data post-training, indicating that the main challenge is not fitting the demonstrated skill itself, but generalizing beyond the post-training instruction coverage.

Table 5: In-distribution Success Counts (20 trials per task).





D.2 Full Qualitative Novel-Prompt Rollout Results

We present full qualitative rollout results of DeLock under OOD settings. Notably, in Food-on-Plate [C], we introduce banana and grapes into the scene, although neither object appears during the SFT stage. We then prompt the policy to “grasp banana” and place it onto the OOD target plate. Despite this combined concept and location shift, the policy is able to follow the novel instruction and successfully complete the task.

Figure 7: Full Qualitative Results with Novel Prompts.

Additionally, in the Mug-on-Plate [S] task, we place the book at the center of the scene and directly prompt the policy with “put book on left plate” without any further fine-tuning. DeLock is able to reach the book and place it on the left plate, whereas the standard fine-tuning baseline produces a largely random trajectory. This result suggests that DeLock can transfer the learned spatial behavior to a novel object specified only through the prompt, without any additional fine-tuning.

Figure 8: Prompted for a Novel Object.




D.3 CPG with an Invalid Positive Prompt

Finally, we test whether CPG succeeds specifically by leveraging the semantic content of the positive prompt. We again consider the Mug-on-Plate [S] task, but replace the intended positive prompt τ+\tau^{+} with the malformed instruction “put mug on write plate”. Since “write” does not specify a meaningful spatial target, the resulting positive condition no longer provides the information needed to steer the policy toward the right plate. In this case, the robot reverts to the post-training bias and places the mug on the left plate. This result suggests that CPG is effective only when the positive prompt provides valid task-relevant semantics; when it succeeds, the resulting behavior is specifically steered by the information encoded in the positive prompt.

Figure 9: CPG with an Invalid Positive Prompt.




D.4 Comparison with a Baseline that Uses Foundation-Model Supervision During Post-Training

We also compare DeLock against a strong baseline, Spatial Forcing [32], which incorporates additional supervision from an external 3D foundation model during post-training to encourage implicit spatial representation alignment. We evaluate both methods on the four simulation tasks in our lock-in benchmark. As shown in Table 6, despite leveraging additional external supervision, Spatial Forcing exhibits very limited adaptation to novel situations across all four tasks, whereas DeLock achieves substantially stronger generalization.

Table 6: Generalization Comparison in Simulation Tasks (20 trials per task).








