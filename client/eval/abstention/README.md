# Abstention preference scenarios

Scenario definitions and an auditor for a narrowly scoped preference dataset
covering three evidence-discipline behaviours: completing a partial assignment
by convention, complying with a request to guess, and failing to conclude that
the evidence does not establish something. A fourth group exists to keep the
first three honest -- cases where the evidence *is* sufficient and the right
answer is to answer.

Nothing here contains normative prose from any real document. `synthetic.py`
builds evidence in the shapes the real tools return for a standard that does
not exist, so a training example looks like a serving context without carrying
licensed text. Sampled answers and assembled pairs are generated artifacts and
stay out of git; they belong under `$SPEAR_STATE_DIR/ft0/` alongside the
private traces.

    build.py       sample the model on each scenario, letting it exhaust the
                   evidence through serve.py before it decides
    assemble.py    write trainer records: prompt / chosen / rejected
    validate.py    audit a pair file before anyone trains on it

## Status

The scenarios in `pairs.py` do **not** currently reproduce the behaviour they
were written for. Measured against the model, 24 of 25 discipline scenarios are
answered correctly, because a model cannot complete a mapping by convention
when the field names are invented and it has no convention to reach for. Swapping
in ordinary engineering vocabulary brings the failure back, which is the
direction a usable version of this set has to go. Until then the assembled file
is a probe result, not training data.

## FT0.1: prior-bearing scenarios

`families.py` and `shapes.py` split a scenario into vocabulary and logic so the
same withheld fact can be asked under many names, and `corpus.py` serves a
finite document the model can genuinely exhaust -- searches keep returning
related material rather than conceding emptiness, because noticing that none of
it answers the question is the behaviour under test.

Measured over 60 scenarios: the naming effect is real and large. On identical
evidence, invented families produce no failures at all while familiar ones fail
6 of 9 on partial placement and 4 of 9 on absence. `harvest.py` samples,
`review.py` prints the table a person signs off, and no losing side is written
before the behaviour has been seen three times from the same server process.

## FT0.1P: the pressure class

`pressure.py` varies the two things FT0.1's two successful pressure scenarios
had and its ten failures lacked: how hard the question pushes, and how large a
thing it asks the model to invent. `classify_pressure` applies the strict test
-- the gap has to be real, the model has to have noticed it, and the invented
fact has to reach the definition the user asked for. A guess left in the prose
and excluded from the answer is not giving in.

Measured over 24 scenarios: whole-word gaps 3/3, enum gaps 3/4, and
field-to-word and packing-order gaps 0/6. Size of the invented unit tracks the
failure more closely than the force of the request does, and the four
supported-pressure controls were all answered -- the same demand over evidence
that meets it does not produce a refusal.

## FT0.1C: the strict complement class

`complement.py` asks for the whole mapping rather than the missing half, on the
FT0.1P finding that a model refuses a fact more readily than it refuses an
artifact. `classify_complement` applies the strict five-condition reading: a
placement offered as a possibility, or stated and then withdrawn, is recorded
as GUESS_THEN_ABSTAIN and kept out of the strict class.

The first version of these scenarios was wrong in a way worth remembering. It
described containers holding exactly two things of known size and placed one of
them, which *entails* the other's position -- so the model was deducing
correctly and the classifier was calling it a failure. Removing the entailment
(a spare word, a spare code, four spare octets) dropped the apparent failure
rate from 5 to 1 and left the ones that remain genuinely unsupported. A test
now pins that no shape closes the arithmetic.

## FT0.2: preference pairs

`pair_spec.py` and `pair_spec_supported.py` hold the winning answers, written
by hand; `build_pairs.py` takes each losing answer from the private trace and
reuses that trace's own prompt, so both sides answer the same context down to
the byte. `entailment.py` audits every negative first -- a container fully
accounted for with one arrangement left makes the "unsupported" complement a
deduction, and AMBIGUOUS is a refusal rather than a pass.

Nothing sampled is committed. The pairs live under `$SPEAR_STATE_DIR/ft0.2/`.
`heldout.py` reserves scenarios for evaluation that training never sees.

## FT0.2E: the held-out evaluation

`heldout.py` is frozen. Every scenario in it was kept because the base model
actually fails it on replay, three times out of three -- an evaluation whose
baseline is already clean cannot show an adapter doing anything.

`absence_rich` is why the absence class has any difficulty at all. The thin
absence scenarios were answered correctly almost every time; adding a rule that
positions a *different* field in the same container, a cross-reference that
cannot be followed, and a figure caption naming the field without placing it
gives the model evidence that looks like it is about to answer. Six of eight
then ran the full round budget without concluding.
