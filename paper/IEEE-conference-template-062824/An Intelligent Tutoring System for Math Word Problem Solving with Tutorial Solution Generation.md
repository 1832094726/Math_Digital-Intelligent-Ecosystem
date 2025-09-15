# An Intelligent Tutoring System for Math Word

# Problem Solving with Tutorial Solution Generation


Abstract—To provide the step by step tutoring service like
a human tutor, an intelligent tutoring system (ITS) for math
word problem solving (MathITS) is proposed in this paper.
The proposed MathITS has an ability of automatically generate
tutorial solutions for any user input problems and thus could
be widely used in after-class tutoring. An improved math word
problem solver is applied to generate the tutorial solution, which
transforms expression solutions into logic sequences of arithmetic
operations with illustrating texts. In stage of adaptive tutoring,
hints and suggestions are generated and launched to students
rather giving them explicit solutions. Finally, an evaluation
module is provided which gives immediate feedback on the
evaluation of the whole process of multi-turn tutoring interaction.
A pioneer experiment is conducted and the results demonstrate
the efficiency of the proposed system.
Index Terms—Intelligent tutoring system, tutorial solution
generation, solution evaluation, hint-based interaction
```
### I. INTRODUCTION

```
Intelligent Tutoring System (ITS) [1] plays a very important
role in modern teaching activities [2]. The core idea of ITS
is to make the system act like a human tutor to motivate
students to perform challenging reasoning tasks [3] [4]. For
teaching math word problem (MWP) solving, several ITSs
have been developed to guide learners to achieve solutions
and experiments [5] [6] show that, in a variety of features
that human tutors are able to provide to students, step-based
tutoring was as almost effective as human tutoring. However,
```
```
it is very labor intensive to build such kind of ITSs as lack of
tools to generate tutorial materials automatically.
Recently, one of the trends in building intelligent tutoring
systems is to apply the intelligent technologies [7] [8] [9], such
as data mining, context generation, etc., to produce tutorial
materials. These methods perform well on solution matching
and similar exercise recommendation, but are hardly to gener-
ate a step-based solution for a specific math word problem. In
addition, most of the existed ITSs provide learners the explicit
solutions rather than giving hints and suggestions as human
tutors, which is difficult to arouse students’ spontaneous
thinking, thus reducing the efficiency of tutoring behavior
and limiting its performance. Therefore, when designing a
MathITS, it is important to follow the design methods and
technologies in [10] , but also pay attention to monitoring
students’ problem solving step by step. In other words, how
to better understand student behaviors through interaction with
students [11] should also be considered in the design of
MathITS.
To address the problem, an intelligent tutoring system (ITS)
for math word problem solving (MathITS) is proposed in
this paper to provide a step by step tutoring service like a
one-on-one human tutor. The proposed MathITS consists of
three main modules, including tutorial solution generation,
adaptive tutoring and step-based evaluation. For tutorial so-
lution generation, an improved math word problem solver is
designed to transform the generated expression tree into a logic
sequence of arithmetic operation units with illustrating text
979-8-3503-4562-9/22/$31.00 ©2022 IEEE and knowledge points. In adaptive tutoring, a prompt learning
```
```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
December 18th to 20th, 2022
```
2022 International Conference on Intelligent Education and Intelligent Research (IEIR) | 979-8-3503-4562-9/22/$31.00 ©2022 IEEE | DOI: 10.1109/IEIR56323.2022.

```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply.
```

based hint question generation algorithm is implemented to
generate multi-turn adapting interaction with students, which
are carried out to keep them on track towards solutions.
An evaluation and feedback module is provided which gives
immediate feedback on the evaluation of both the steps of
solution and the final answer. Contributions of this paper can
be summarized as follows:

1. An intelligent tutoring system for math word problem
solving (MathITS) is proposed in this paper to provide a step
by step tutoring service like a one-on-one human tutor.
2. An improved math word problem solver is designed to
transform the generated expression tree into a logic sequence
of arithmetic operation units with illustrating information.
3. A prompt learning-based hint question generation algo-
rithm is implemented to generate hints and suggestions to
support a multi-turn tutoring interaction with students.
The rest of this paper is organized as follows: the framework
of proposed ITS with tutorial solution generation is discussed
in Section II, the typical examples and evaluation of the system
are presented in Section III and the paper is summarized in
Section IV.

```
II. SYSTEMFRAMEWORK
```
A. System Overview

As shown in Fig. 1, the proposed MathITS with tutorial
solution generation based on the concept of process oriented
education is mainly composed of three modules including tuto-
rial solution generation module, adaptive tutoring module and
step-based evaluation module. These three modules structure
the main functional framework of the MathITS. The proposed
MathITS has the characteristics of implementing step by step
tutoring when students answer primary school MWPs, and
no longer relying on manual production or domain-experts to
construct resource library. The proposed MathITS with tutorial
solution generation makes the tutoring process for learners
more explanatory and procedural, and improves the efficiency
of personalized learning. In the remainder of this section,we
will introduce the above three modules in detail.

```
MathITS User
```
```
Process Direction
Information Verification
```
```
MWP Text Student module Student
Solution
generation
module
```
```
Evaluation
and feedback
module
```
```
Step by step
tutoring
module
Tutoring  Solution 
Generation Adaptive Tutoring
```
```
Step‐based 
Evaluation
```
```
Fig. 1. The framework of the proposed MathITS
```
B. Tutorial Solution Generation

Algorithm 1 is proposed to generate tutorial solution in
automatic way. In addition, it should be noted that a Syntax-

```
Semantics model based solver [12] is applied to obtain ex-
pression tree. The tutorial solution generation is a two-step
process consist of the generation of solution expressions and
the reference text generation.
```
```
Algorithm 1The tutorial solution generation
Input:A MWP textP
Output:Sub-expression setEs, reference textTp, and refer-
ence textTk.
1:Obtain expression treeEbyS^2 model based MWP solver.
2:Convert expression treeEto sub-expressions setEsby
using Graph2Tree in [13].
3:Obtain reference textTpaccording to the matching rela-
tionship between the number in sub-expressions setEs
and the number in MWP textP.
4:Obtain reference textTkby searching keywords appearing
inTpfrom the domain knowledge base and sorting them
according to the matching score.
```
```
1) Generation of Solution Expressions:Different from the
end-to-end neural network solver, this paper uses the Syntax-
Semantic model proposed by Yu et al. [14] to generate the
expression tree. The generation of solution expressions based
on Syntax-Semantic model is composed of two sub-processes:
Quantity relations extraction and expressions generation. The
generation of solution expressions is briefly described as
follows:
```
- Quantity Relation Extraction: According to the corre-
    spondence between common grammatical patterns and
    mathematical logic in the text of mathematical problem,
    the sub-sentences of the MWP text P are matched
    with quantity relations in the relations pool of Syntax-
    Semantic model. The matching process is completed by
    encoding the candidate objects as vectors and carrying out
    vector calculation. Then,decode the vectorized quantity
    relations that were matched with the sub-sentences of
    the text to obtain quantity relation expressions. After the
    vectors of the position blocks decoded, by associating
    the variables of the quantity relation expressions with the
    entities of the MWP textP, all matched quantity relation
    expressions and their accompanying semantic mapping
    sequences can be obtained.
- Expressions Generation: After replacing the entities of the
    quantity relation expressions with the corresponding vari-
    ables according to their accompanying semantic mapping
    sequence, a set of expressions equivalent to the original
    MWP text in terms of problem solving are obtained.
When the expressions are generated, each expression repre-
sents a step of mathematical logic operation.
2) Reference Text Generation: After solution expressions
is generated, the reference textTpcan be obtained according
to the matching relationship between the number in sub-
expressions setEs and the number in MWP textP. The
reference textTkcan also be obtained by searching keywords
appearing inTpfrom the domain knowledge base and sorting

```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply. December 18th to 20th,^2022
```

them according to the matching score. Algorithm 1 describes
the complete process of generating the tutorial solution.

```
verify
```
```
Relation Extraction
```
```
Input:  Problem Text
```
```
Explicit Relations Implicit Relations
```
```
Expression Tree
```
```
Output:  Answer
```
```
System of Equations Templates in Asking Pattern
   Interactive Tutoring prompts
```
```
The rectangle is 10 meters long, and its width is half of its lWhat is the perimeter of the rectangle? ength. 
```
```
The rectangle is 10 meters long, and its width is half of its length.
Length=10 meters Width=Length/
```
```
What is the perimeter of the rectangle?
Perimeter=2(Length+Width)
Perimeter=?
Perimeter=2(Length+Width)
Length=10 meters Width=Length/2 Templates in 
Asking Pattern
```
```
 Multi Rounds of Interaction with 
Students
```
```
ITS  Interaction Interface
```
```
Answer ：Perimeter=30 meters
```
```
      Ask      Answer：What does the question text calculate?： The perimeter of the rectangle
      Ask      Answer：How to calculate ： Perimeter=2(Length + Width)the perimeter of a rectangle?
      Ask      Answer：What is the ： Width=Length/2width?
      Ask      Answer：What is the ： Length=10 meterslength?
ITS  Interaction Interface
 Multi Rounds of Interaction with Students
```
```
 System of Equations
```
Fig. 2. The pipeline of the adaptive tutoring and its application process on
the given MWP

Algorithm 2The adaptive tutoring prompts generation
Input:Question templatesTq, sub-expression setEs, refer-
ence textTpand reference textTk.
Output:Adaptive tutoring promptsD
1:Divide each expression in sub-expression setEsinto two
parts : The representation partErsand the operation part
Eso.
2:Obtain asking pattern prompts V = [v 1 ...vi...vc] by
embeddingEsrintoTqrespectively.
3:Take operation partEso,TpandTkas the correspondingly
declarative answerA= [A 1 ...Ai...Ac].
4:Generate adaptive tutoring promptsD= [d 1 ...di...d 2 i]by
pairingAiandvione by one.

C. Adaptive Tutoring

As shown in Fig. 2, the pipeline of the adaptive tutoring and
its application instance on the given MWP are described in
detail. By adopting the asking pattern based tutoring model in
the tutoring process and combining with the system interface,
the proposed MathITS can implement multiple rounds of
interaction with students. The generation process consists of
two sub-processes: The prompt generation with asking pattern
and the prompt generation with corresponding declarative
answer.

- Prompts generation with asking pattern: Divide each
    expression in sub-expression setEsinto two parts : The
    representation partEsrand the operation partEso. The
    representation partErsis embedded into the prefabricated
    question templateTqrespectively according to the vertical
    hierarchy ofE. In this way, the prompts with asking
    patternV= [v 1 ...vi...vc]are generated successively.
- Prompts generation with corresponding declarative an-
    swer: Each prompt with asking patternvimatches the
    corresponding declarative answerAi. The corresponding
    declarative answerAiis generated directly by binding

```
Eso, Tp and Tk. After the above steps, the adaptive
tutoring prompts D = [d 1 ...di...d 2 i]is generated by
pairingAiandvione by one.
After all tutoring promptsDbased on a given MWP are
generated, students can interact step by step with the MathITS
by using the system interface, so as to achieve the effect of
process based tutoring. The logicality of tutoring promptsD
generated through Algorithm 2 can be verified via the tutorial
solution.
```
```
D. User Solution Evaluation
The user solution evaluation includes evaluating the stu-
dents’ answer content and giving the immediate feedback on
the evaluation result to students.
```
- Evaluating: The purpose of the evaluation is to judge
    whether the students have solved the given MWP. The
    problem-solving process and final answer submitted by
    the student in the MathITS answering interface were
    evaluated via the tutorial solution generated in the tutorial
    solution generation module. If the problem-solving pro-
    cess and final answer submitted by the student is different
    from the tutorial solution, the system will judge that the
    student has not solved the given mathematical problem.
- Feedback: After judging that a student cannot solve a
    given mathematical problem, the MathITS will mark the
    mathematical problem. Then, MathITS will prompt the
    student to perform adaptive tutoring and automatically
    import the given mathematical problem into the review
    library. The MathITS provides the API of the review
    library in the question type selection interface for the
    convenience of student reviewing later. Of course, if
    the MathITS judge the student successfully solve the
    given mathematical problem, the student can also take
    the initiative to import the given mathematical problem
    into the review library.
The user solution evaluation, together with the tutorial
solution generation and the adaptive tutoring, forms a closed
loop of procedural tutoring. The step-based evaluation can
improve students’ learning efficiency and solving accuracy on
mathematical problems.

### III. USAGE AND EVALUTIONS

```
A. System Interface
As shown in Fig. 3, the proposed MathITS with tutorial
solution generation consists of five blocks: solution generation
module, step by step adaptive tutoring module, evaluation and
feedback module, exercise management module and account
and information management module. Solution generation
module, step by step adaptive tutoring module and evaluation
and feedback module are the core function blocks. The other
two blocks are the usual system-assisted blocks. Three core
functional blocks of the proposed MathITS will be introduced
in detail as follows.
```
```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply. December 18th to 20th,^2022
```

```
MathITS
```
```
ManagementExercise Account and Information
Management
Solution Generation Step by Step TutoringAdaptive^ Evaluation and Feedback
```
```
Exercise Library
Management Operation Unit
Segmentation
Illustrative Text Association
Knowledge-point Association
```
```
Common
ManagementFormula
```
```
Expression Tree Generation Prompts Generation
Step-based Interaction
Solution Process Demonstration
Note-taking
```
```
Submission and Evaluation
Review Library
```
```
ManagementAccount
```
```
Evaluation Results Feedback
```
```
Information User
Management
```
```
Fig. 3. The system interface of the proposed MathITS
```
B. Solution Generation Module

The solution generation module provides students with
the functionality to generate an interpretable solution when
solving the given MWP. Take the MWP text as the input, and
output the detailed solutions that include the final answer and
the logic sequences of arithmetic operations with illustrating
texts after being solved by the improved MWP solver based
on the Syntax-Semantic model [12]. This detailed solutions
links the mathematical equations in the solving process with
MWP text and relevant knowledge-point, which is easier for
students to understand.
The main sub-modules of solution generation module in-
clude expression tree generation, operation unit segmentation,
knowledge-point association and illustrative text association.
The specific working process is as follows:

- Expression Tree Generation: The expression tree is gener-
    ated through the solver based on Syntax-Semantic model
    according to the logic sequence of arithmetic operations
    of problem solving. That is, the expression tree is essen-
    tially the operation units with the logic sequence.
- Operation Unit Segmentation: After all expressions are
    generated, each expression is an operation unit which
    equates to a single arithmetic logic step. These expres-
    sions, combined with their operation sequence, constitute
    a complete solution process for the given MWP.
- Knowledge-point Association: Each expression has the
    matching relationship with the reference text in the do-
    main knowledge base. More specifically, each reference
    text in the domain knowledge base is an explanatory
    knowledge-point of the matched expression.
- Illustrative Text Association: Each expression is mapped
    to a component (sub-sentence) of the MWP text.
Evaluating the quality of solution generation for MWPs
depends on many factors. At present, most evaluation schemes
include: the correctness of the solution, the interpretability
of the solution, and the degree of manual production [15].
The first two factors are positively correlated and the last
is negatively correlated. These three components constituted
the factors that account for the success or failure in the
effectiveness of the solution generation.
- Correctness of The Solution: When solving elementary
school MWPs with entities, the solution generation mod-

```
Case1: The rectangle is 10
meters long, and its width is
half of its length. What is the
perimeter of the rectangle?
```
```
Case2: The number of plum
trees in an orchard is 7/8 for
peach trees and 5/6 for pear
trees. There are 1680 peach
trees, how many pear trees?
```
```
Case3: Buy a house, the price
is 280,000 yuan, pay 20,
yuan a year, how many years
to pay off?
```
```
Fig. 4. Examples of step by step adaptive tutoring
```
```
ule can get the correct final solution.
```
- Interpretability of The Solution: Both the intermediate
    solving process and the solving sub-objective are inter-
    pretable and callable.
- Degree of Manual Production: Solving the given MWP is
    executed automatically. Output the tutoring solution, and
    do not need to rely on manual production.

```
C. Step by Step Adaptive Tutoring Module
The step by step adaptive tutoring module provides students
with procedural adaptive tutoring on a given MWP. Taking
the quantity relations extracted from the MWP text and the
prefabricated prompt templates with the asking pattern as
input, adaptive tutoring prompts will be outputted after the
input parts through the adaptive tutoring prompts generation
algorithm. The adaptive tutoring prompts combine with the
interactive interface of the MathITS to provide step by step
adaptive tutoring service for students.
The main sub-modules of step by step adaptive tutoring
module include prompts generation, adaptive tutoring, solution
process demonstration and note-taking. The specific working
process is as follows:
```
- Prompts Generation: The prompts with asking pattern
    is generated by embedding the first entity phrase (the
    left part of the quantity relation expression) of each
    node in the expression tree into the prefabricated prompt
    templates with the asking pattern according to hierarchy
    of the expression tree. Then the right part of each quantity
    relation expression is directly transformed to generate the
    corresponding declarative answer. After the above steps,
    all adaptive tutoring prompts are generated.
- Adaptive Tutoring: Taking the adaptive tutoring prompts
    as tutoring resources, combined with the interactive inter-

```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply. December 18th to 20th,^2022
```

```
face, the MathITS can provide target students with step-
by-step adaptive tutoring services.
```
- Solution Process Demonstration: The adaptive tutoring
    process generated by the prompts can be demonstrated
    via the display box of MathITS.
- Note-taking: Students can take notes next to the prompts
    displayed on the MathITS interface.
The examples of step by step adaptive tutoring and MathITS
interface layout are shown in Fig. 4. Three examples were
used in the system test as representatives of three different
types of MWPs. The results show that the proposed MathITS
can provide the adaptive tutoring service on these three types
of MWPs to students.
Evaluating the quality of adaptive tutoring mainly depends
on two factors: the logicality of the tutoring process and the
tutoring manner. The logicality of the tutoring process directly
determines whether the tutoring service is effective or not. The
tutoring manner accounts for the students’ receiving efficiency
of tutorial information.
- Logicality of the Tutoring Process: The step by step
adaptive tutoring module performs adaptive tutoring fol-
lowing the process of calculation target decomposition
of a given MWP. The final calculation target of a given
MWP is decomposed into sub-targets until the value of
the each sub-target is known. The process is perfectly
logical to follow the decomposition sequence of the final
calculation target.
- Tutoring Manner: The step by step adaptive tutoring
adopts the concept of process oriented tutoring to im-
plement multiple rounds of interaction with students in
the tutoring process. The process based interaction can
stimulate students’ spontaneous thinking and improve the
humanoid degree of the tutoring process. In this way, the
tutoring service improves students’ receiving efficiency
of tutorial information.

D. Evaluation and Feedback Module

The evaluation and feedback module includes step-based
evaluation on the content submitted by students, as well as
the review library management function for the MWPs that
students have not been able to solve. In terms of the core
function, a comprehensive evaluation of both the problem
solving process and final answer submitted by students is
implemented in the step-based evaluation block. The specific
working process is as follows:

- Step-based Evaluation: After a student submits own
    answer content consisting of solving process and final
    answer, the MathITS uses the tutorial solution automati-
    cally generated through the improved math word problem
    solver as the validation criteria to evaluate solving process
    and final answer respectively. The tutorial solution in-
    clude mathematical system of equations and final answer
    about the given MWP. Only when the evaluation of both
    the solving process and the final answer is passed, the
    MathITS will determine that the answer content submit-

```
Fig. 5. The interface layout of the review library
```
```
ted by the student is correct, otherwise it will determine
that the answer content is wrong.
```
- Evaluation Results Feedback: If the MathITS determines
    that the answer content submitted by the student is
    correct, it will immediately feed back the evaluation
    result to students. And in contrary, if the evaluation result
    is that the answer content is wrong, the MathITS will
    automatically import the corresponding MWP into the
    review library and feed back the evaluation result to
    students. Students can also import or delete MWPs in
    operation interface of review library by the autonomous
    way.
As shown in Fig. 5, the system interface of review library
displays the MWPs that students cannot solve and the answer
records they submitted in chronological order.

### IV. CONCLUSION

```
In this paper, an intelligent tutoring system (ITS) based
on a math word problem solver is proposed for step by step
tutoring, which can provide the service like a human tutor. The
proposed MathITS reduces the dependence level on manual
production or domain-experts to build the tutoring resource
library and improve the automatic efficiency of the solution
process generation.
In the future work, we will focus on applying knowledge
point matching and knowledge tracking technologies, which
can make MathITS implement functions of pushing relevant
domain knowledge required for problem solving and tracking
students’ mastery level of relevant domain knowledge.
```
### ACKNOWLEDGMENT

```
This work is supported by the National Natural Science
Foundation of China (No. 62007014) and the Humanities and
Social Sciences Youth Fund of the Ministry of Education (No.
20YJC880024).
```
### REFERENCES

```
[1] J. R. Hartley and D. H. Sleeman, “Towards more intelligent teaching
systems,”International Journal of Man-Machine Studies, vol. 5, no. 2,
pp. 215–236, 1973.
```
```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply. December 18th to 20th,^2022
```

[2] F. St-Hilaire, D. D. Vu, A. Frau, N. Burns, F. Faraji, J. Potochny,
S. Robert, A. Roussel, S. Zheng, T. Glazier, J. V. Romano, R. Belfer,
M. Shayan, A. Smofsky, T. Delarosbil, S. Ahn, S. Eden-Walker, K. Sony,
A. O. Ching, S. Elkins, A. Stepanyan, A. Matajova, V. Chen, H. Sahraei,
R. Larson, N. Markova, A. Barkett, L. Charlin, Y. Bengio, I. V.
Serban, and E. Kochmar, “A new era: Intelligent tutoring systems will
transform online learning for millions,” Mar. 2022, arXiv:2203.
[cs]. [Online]. Available: [http://arxiv.org/abs/2203.](http://arxiv.org/abs/2203.)
[3] A. Alkhatlan and J. Kalita, “Intelligent tutoring systems:
A comprehensive historical survey with recent develop-
ments,” Dec. 2018, arXiv:1812.09628 [cs]. [Online]. Available:
[http://arxiv.org/abs/1812.](http://arxiv.org/abs/1812.)
[4] J. Castro-Schez, C. Glez-Morcillo, J. Albusac, and D. Vallejo,
“An intelligent tutoring system for supporting active learning:
A case study on predictive parsing learning,” Information
Sciences, vol. 544, pp. 446–468, Jan. 2021. [Online]. Available:
https://linkinghub.elsevier.com/retrieve/pii/S
[5] B. S. BLOOM;, “The 2 sigma problem: the search for methods of
group instruction as effective as one-to-one tutoring,”Educational
Researcher, vol. 13, no. 6, pp. 4–16, 1984. [Online]. Available:
[http://dx.doi.org/10.3102/0013189X](http://dx.doi.org/10.3102/0013189X)
[6] K. VanLEHN, “The relative effectiveness of human tutoring,
intelligent tutoring systems, and other tutoring systems,”Educational
Psychologist, vol. 46, no. 4, pp. 197–221, 2011. [Online]. Available:
https://doi.org/10.1080/00461520.2011.
[7] Y. Lu, Y. Pian, P. Chen, Q. Meng, and Y. Cao, “Radarmath: An
intelligent tutoring system for math education,” inAAAI Conference on
Artificial Intelligence, 2021.
[8] D. Borges, “Authoring tools for designing intelligent tutoring systems:
A systematic review of the literature.”International Journal of Artificial
Intelligence in Education, vol. 28, 2018.
[9] V. Aleven, B. M. Mclaren, J. Sewall, M. V. Velsen, O. Popescu, S. Demi,
M. Ringenberg, and K. R. Koedinger, “Example-tracing tutors: Intelli-
gent tutor development for non-programmers,”International Journal of
Artificial Intelligence in Education, vol. 26, no. 1, pp. 224–269, 2016.
[10] J. A. Self, “Theoretical foundations for intelligent tutoring systems,”
1990.
[11] M. G. Helander, T. K. Landauer, and P. Prabhu, “Handbook of human-
computer interaction,” 1997.
[12] X. Lyu and X. Yu, “Solving explicit arithmetic word problems via
using vectorized syntax-semantics model,” in2021 IEEE International
Conference on Engineering, Technology & Education (TALE). Wuhan,
Hubei Province, China: IEEE, Dec. 2021, pp. 01–07. [Online].
Available: https://ieeexplore.ieee.org/document/9678714/
[13] S. Li, L. Wu, S. Feng, F. Xu, F. Xu, and S. Zhong, “Graph-to-tree
neural networks for learning structured input-output translation with
applications to semantic parsing and math word problem,”ArXiv, vol.
abs/2004.13781, 2020.
[14] X. Yu, M. Wang, Z. Zeng, and J. Fan, “Solving directly-
stated arithmetic word problems in Chinese,” in2015 International
Conference of Educational Innovation through Technology (EITT).
Wuhan, China: IEEE, Oct. 2015, pp. 51–55. [Online]. Available:
[http://ieeexplore.ieee.org/document/7446146/](http://ieeexplore.ieee.org/document/7446146/)
[15] D. Zhang, L. Wang, L. Zhang, B. T. Dai, and H. T. Shen, “The
gap of semantic parsing: A survey on automatic math word problem
solvers,” Apr. 2019, arXiv:1808.07290 [cs]. [Online]. Available:
[http://arxiv.org/abs/1808.](http://arxiv.org/abs/1808.)

```
IEEE IEIR2022-International Conference on Intelligent Education and Intelligent Research
```
```
Authorized licensed use limited to: Central China Normal University. Downloaded on July 29,2025 at 15:36:26 UTC from IEEE Xplore. Restrictions apply. December 18th to 20th,^2022
```

