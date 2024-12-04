import sys
import os

import numpy as np
import pandas
from sklearn.metrics import roc_curve

# Fix tandem detection cost function (t-DCF) parameters
Pspoof = 0.05
cost_model = {
    "Pspoof": Pspoof,  # Prior probability of a spoofing attack
    "Ptar": (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    "Pnon": (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    "Cmiss": 1,  # Cost of ASV system falsely rejecting target speaker
    "Cfa": 10,  # Cost of ASV system falsely accepting nontarget speaker
    "Cmiss_asv": 1,  # Cost of ASV system falsely rejecting target speaker
    "Cfa_asv": 10,  # Cost of ASV system falsely accepting nontarget speaker
    "Cmiss_cm": 1,  # Cost of CM system falsely rejecting target speaker
    "Cfa_cm": 10,  # Cost of CM system falsely accepting spoof
}


def produce_evaluation_file(utt_ids, eval_scores, cm_path, save_path):
    if cm_path is None:
        with open(save_path, "w") as f:
            for utt_id, score in zip(utt_ids, eval_scores):
                f.write("{} {}\n".format(utt_id, score))
        return

    with open(cm_path, "r") as f:
        label_lines = f.readlines()

    labels = {line.strip().split(" ")[1]: line for line in label_lines}

    with open(save_path, "w") as f:
        for utt_id, score in zip(utt_ids, eval_scores):
            _, _, _, src, key = labels[utt_id].strip().split(" ")
            f.write("{} {} {} {}\n".format(utt_id, src, key, score))
    print("Scores saved to {}".format(save_path))


# calculate min t-DCF and EER for ASVspoof 2019 LA
def calculate_tDCF_EER_19LA(cm_scores_file, asv_score_file, output_file, printout=True):
    # Replace CM scores with your own scores or provide score file as the
    # first argument.
    # cm_scores_file =  'score_cm.txt'
    # Replace ASV scores with organizers' scores or provide score file as
    # the second argument.
    # asv_score_file = 'ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float64)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == "target"]
    non_asv = asv_scores[asv_keys == "nontarget"]
    spoof_asv = asv_scores[asv_keys == "spoof"]

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == "bonafide"]
    spoof_cm = cm_scores[cm_keys == "spoof"]

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    attack_types = [f"A{_id:02d}" for _id in range(7, 20)]
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer(bona_cm, spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, *_ = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(
        bona_cm,
        spoof_cm,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        cost_model,
        print_cost=False,
    )

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write("\nCM SYSTEM\n")
            f_res.write(
                "\tEER\t\t= {:8.9f} % "
                "(Equal error rate for countermeasure)\n".format(eer_cm * 100)
            )

            f_res.write("\nTANDEM\n")
            f_res.write("\tmin-tDCF\t\t= {:8.9f}\n".format(min_tDCF))

            f_res.write("\nBREAKDOWN CM SYSTEM\n")
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f"\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type}\n"
                )
            f_res.write("\n")
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF


# calculate min t-DCF and EER for ASVspoof 2021 LA
def calculate_tDCF_EER_21LA(score_file, truth_dir, output_file=None):
    phase = "eval"
    # Load organizers' ASV scores
    asv_key_file = os.path.join(truth_dir, "ASV/trial_metadata.txt")
    asv_scr_file = os.path.join(truth_dir, "ASV/ASVTorch_Kaldi/score.txt")
    cm_key_file = os.path.join(truth_dir, "CM/trial_metadata.txt")
    asv_key_data = pandas.read_csv(asv_key_file, sep=" ", header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=" ", header=None)[
        asv_key_data[7] == phase
    ]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == "target"
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == "nontarget"
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == "spoof"

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    cm_data = pandas.read_csv(cm_key_file, sep=" ", header=None)
    submission_scores = pandas.read_csv(
        score_file, sep=" ", header=None, skipinitialspace=True
    )

    if len(submission_scores) != len(cm_data):
        print(
            "CHECK: submission has %d of %d expected trials."
            % (len(submission_scores), len(cm_data))
        )
        exit(1)

    # check here for progress vs eval set
    cm_scores = submission_scores.merge(
        cm_data[cm_data[7] == phase], left_on=0, right_on=1, how="inner"
    )
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100 * eer_cm)
    print(out_data, end="")

    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance(
        cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True
    )

    if min_tDCF2 < min_tDCF:
        print(
            "CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking"
            % (min_tDCF, min_tDCF2)
        )

    if min_tDCF == min_tDCF2:
        print(
            "WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?"
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("min_tDCF: %.4f\n" % min_tDCF)
            f.write("eer: %.2f%\n" % (100 * eer_cm))

    return min_tDCF


def calculate_EER_21DF(score_file, cm_key_file, output_file=None):
    phase = "eval"
    cm_data = pandas.read_csv(cm_key_file, sep=" ", header=None)
    submission_scores = pandas.read_csv(
        score_file, sep=" ", header=None, skipinitialspace=True
    )
    if len(submission_scores) != len(cm_data):
        print(
            "CHECK: submission has %d of %d expected trials."
            % (len(submission_scores), len(cm_data))
        )
        exit(1)

    if len(submission_scores.columns) > 2:
        print(
            "CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces."
            % len(submission_scores.columns)
        )
        exit(1)

    cm_scores = submission_scores.merge(
        cm_data[cm_data[7] == phase], left_on=0, right_on=1, how="inner"
    )  # check here for progress vs eval set
    bona_cm = cm_scores[cm_scores[5] == "bonafide"]["1_x"].values
    spoof_cm = cm_scores[cm_scores[5] == "spoof"]["1_x"].values
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100 * eer_cm)
    print(out_data)
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("eer: %.2f\n" % (100 * eer_cm))
    return eer_cm


def calculate_EER_IntheWild(score_file, label_file, output_file=None):
    scores = []
    with open(score_file, "r") as f:
        for line in f:
            _, score = line.strip().split(" ")
            scores.append(float(score))

    labels = []
    with open(label_file, "r") as f:
        for line in f:
            _, utt_id, _, _, label = line.strip().split(" ")
            if label == "bonafide":
                labels.append(1)
            else:
                labels.append(0)

    fpr, tpr, threshold = roc_curve(labels, scores)
    # plt.show()
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    print("eer: %.2f" % (100 * eer))
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("eer: %.2f\n" % (100 * eer))
    return eer, eer_threshold


def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5] == "bonafide"]["1_x"].values
    spoof_cm = cm_scores[cm_scores[5] == "spoof"]["1_x"].values

    if invert == False:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if invert == False:
        tDCF_curve, _ = compute_tDCF(
            bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False
        )
    else:
        tDCF_curve, _ = compute_tDCF(
            -bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False
        )

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(
    bonafide_score_cm,
    spoof_score_cm,
    Pfa_asv,
    Pmiss_asv,
    Pmiss_spoof_asv,
    cost_model,
    print_cost,
):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """

    # Sanity check of cost parameters
    if (
        cost_model["Cfa_asv"] < 0
        or cost_model["Cmiss_asv"] < 0
        or cost_model["Cfa_cm"] < 0
        or cost_model["Cmiss_cm"] < 0
    ):
        print("WARNING: Usually the cost values should be positive!")

    if (
        cost_model["Ptar"] < 0
        or cost_model["Pnon"] < 0
        or cost_model["Pspoof"] < 0
        or np.abs(cost_model["Ptar"] + cost_model["Pnon"] + cost_model["Pspoof"] - 1)
        > 1e-10
    ):
        sys.exit(
            "ERROR: Your prior probabilities should be positive and sum up to one."
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            "ERROR: you should provide miss rate of spoof tests against your ASV system."
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit("ERROR: Your scores contain nan or inf.")

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit("ERROR: You should provide soft CM scores - not binary decisions")

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = (
        cost_model["Ptar"]
        * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    )
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            "You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?"
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print(
            "t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n".format(
                bonafide_score_cm.size, spoof_score_cm.size
            )
        )
        print("t-DCF MODEL")
        print(
            "   Ptar         = {:8.5f} (Prior probability of target user)".format(
                cost_model["Ptar"]
            )
        )
        print(
            "   Pnon         = {:8.5f} (Prior probability of nontarget user)".format(
                cost_model["Pnon"]
            )
        )
        print(
            "   Pspoof       = {:8.5f} (Prior probability of spoofing attack)".format(
                cost_model["Pspoof"]
            )
        )
        print(
            "   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)".format(
                cost_model["Cfa_asv"]
            )
        )
        print(
            "   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)".format(
                cost_model["Cmiss_asv"]
            )
        )
        print(
            "   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)".format(
                cost_model["Cfa_cm"]
            )
        )
        print(
            "   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)".format(
                cost_model["Cmiss_cm"]
            )
        )
        print(
            "\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)"
        )

        if C2 == np.minimum(C1, C2):
            print(
                "   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n".format(C1 / C2)
            )
        else:
            print(
                "   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n".format(C2 / C1)
            )

    return tDCF_norm, CM_thresholds
