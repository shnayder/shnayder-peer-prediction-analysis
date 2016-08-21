# shnayder-peer-prediction-analysis

This repo contains code for analyzing peer prediction algorithms on data from peer assessment in MOOCs. It also includes a summarized version of that dataset.

For more details about the dataset and the analyses done with it, see [Victor's dissertation](https://www.eecs.harvard.edu/~shnayder/papers/dissertation.pdf), or the [HCOMP paper](https://www.eecs.harvard.edu/~shnayder/papers/shnayder-hcomp16.pdf).

# Summarized dataset

Thanks to [edX](https://www.edx.org) for allowing me to publish this summarized dataset of joint distributions of pairs of scores on peer assessment questions. The underlying individual responses are sensitive, and cannot be shared.

Data format, in `data/joint-distributions.json`:

    [((course_id, problem_id, criterion_id), joint_distribution_matrix), ...]
      
Example:
```
[
    [
        [
            0, 
            0, 
            0
        ], 
        [
            [
                0.0339095744680851, 
                0.014960106382978724, 
                0.022938829787234043, 
                0.02992021276595745
            ], 
            [
                0.014960106382978724, 
                0.041223404255319146, 
                0.06216755319148936, 
                0.059175531914893616
            ], 
            [
                0.022938829787234043, 
                0.06216755319148936, 
                0.09840425531914894, 
                0.12300531914893617
            ], 
            [
                0.02992021276595745, 
                0.059175531914893616, 
                0.12300531914893617, 
                0.20212765957446807
            ]
        ]
    ], 
...
```

`course_id`s are globally unique. `problem_id`s are globally unique. `criterion_id` are unique within a particular `(course_id, problem_id)` pair. For example, `course_id` 0 may be "MyUniversityX Intro to Computer Science", `problem_id` 0 may be the peer assessment question "Assess your peers' Hello World programs.", and criteria may be "code style", "correctness", "adequate of comments", etc, providing a rubric for each.

The joint distributions are symmetric with pairs of reports (i,j) and (j,i) treated equally and split between those two entries. If the matrix is `n-by-n`, this means that there were `n` possible score options, and the number in (i,j) is the probability of that pair of reported scores when two random peers assessed the same submission (by another student).

# Code

**Disclaimer:** This was written for myself, for data analysis and simulation for my dissertation. Some parts may be obsolete, as things changed over time.  Other parts may rely on the raw peer assessment data I analyzed while at edX, which cannot be shared. I hope that sharing this gives people a chance to check my work, or get inspiration for your own analyses. It's probably not directly usable without modification though...

A mini-guide:

- The ipython notebooks in `notebooks/` explore the data and generate various graphs. They won't run without tweaking paths, etc. 
- The notebooks use various functions and classes from `model`. 
- The notebooks have a variety of notes-to-self as inline text. Some of these may be out of date, if I figured out I misinterpreted something. I'm leaving them in as they may help others follow what I was thinking at the time.

If you have questions, see references to code that's missing (I had a bunch of other older files that I didn't include here to reduce confusion and startup time) or see what looks like a bug, let me know!
