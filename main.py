import os
import argparse
import logging
import time
import math
from datetime import timedelta
import platform

import numpy as np

import jax.numpy as jnp
from jax import jit, lax, device_put
from functools import partial

if 'COLAB_TPU_ADDR' in os.environ:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()

# Helper functions defined by us in their
# respective files
from config import config
import saveload
from log_helper import init_logger

# controls precision globally
jnp_precision = jnp.float32


def is_known_mersenne_prime(p):
    """Returns True if the given Mersenne prime is known, and False otherwise."""
    primes = frozenset([2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091,
                       756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933])
    return p in primes


def is_prime(n):
    """Return True if n is a prime number, else False."""
    return n >= 2 and not any(
        n % p == 0 for p in range(2, int(math.sqrt(n)) + 1))


def main():
    # Read settings and program arguments
    parser = argparse.ArgumentParser(description='TensorPrime')
    # config.getSettings()

    # In order to add more arguments to the parser,
    # attempt a similar declaration to below.
    # Anthing without a dash is becomes ordinal
    # and required
    parser.add_argument('--version', action='version',
                        version='TensorPrime 1.0')
    parser.add_argument("-p", "--prime", "--prp", type=int,
                        help="Run PRP primality test of exponent and exit")
    parser.add_argument(
        "--ll", type=int, help="Run LL primality test of exponent and exit")
    # parser.add_argument("--bench", action="store_true", help="perform testing etc")
    parser.add_argument("--iters", type=int,
                        help="Run test for this many iterations and exit")
    parser.add_argument("-f", "--fft", "--fftlen", "--sig",
                        "--siglen", type=int, help="FFT/Signal length")
    parser.add_argument("--shift", type=int,
                        help="Number of bits to shift the initial seed")
    parser.add_argument("--prp_base", type=int, default=3,
                        help="PRP base, Default: %(default)s")
    parser.add_argument("--prp_residue", type=int, choices=range(1, 6),
                        default=1, help="PRP residue type, Default: %(default)s")
    parser.add_argument("--proof_power", type=int, choices=range(1, 13), default=8,
                        help="Maximum proof power, every lower power halves storage requirements, but doubles the certification cost, Default: %(default)s")
    parser.add_argument("--proof_power_mult", type=int, choices=range(1, 5),
                        help="Proof power multiplier, to simulate a higher proof power by creating a larger proof file")
    parser.add_argument("-w", "--dir", "--workdir", default=".",
                        help="Working directory with the work, results and local files, Default: %(default)s (current directory)")
    parser.add_argument("-i", "--workfile", default="worktodo.txt",
                        help="Work File filename, Default: '%(default)s'")
    parser.add_argument("-r", "--resultsfile", default="results.txt",
                        help="Results File filename, Default: '%(default)s'")
    parser.add_argument("-l", "--localfile", default="local.ini",
                        help="Local configuration file filename, Default: '%(default)s'")
    parser.add_argument("--resume", type=int, default=-1,
                        help="Savefile/Checkpoint to resume from. Most recent is 0, Default %(default)s")
    parser.add_argument("-x", "--output_iter", type=int, default=10000,
                        help="Output/report every this many iterations, Default %(default)s iterations")
    parser.add_argument("--save_count", type=int, default=2,
                        help="Number of savefiles/checkpoints to keep (-1 to keep all), Default %(default)s")
    parser.add_argument("-c", "--save_iters", type=int, default=100000,
                        help="Write savefile/checkpoint every this many iterations, Default %(default)s iterations")
    parser.add_argument("--error", action="store_false", default=True,
                        help="Do Round off error (ROE) checking, Default %(default)s")
    # Prime95/MPrime: if near FFT length limit 1/else 128 iterations (ErrorCheck), if near FFT length limit 0.421875/else 0.40625 (MaxRoundoffError)
    # Mlucas: 1 iterations, 0.40625 warn/0.4375 error,
    # CUDALucas: 100 iterations (ErrorIterations, must have form k*10^m for k = 1, 2, or 5),
        # 40 (-e, ErrorLimit, must be 1-47), thus (ErrorLimit - ErrorLimit / 8.0 * log(ErrorIterations) / log(100.0)) / 100.0 = 0.35 (up to 0.41125)
    # [print("ErrorLimit:", i, "ErrorIterations:", j, "ROE:", (i - i / 8.0 * math.log(j) / math.log(100.0)) / 100.0) for i in range(1, 48) for j in (k*(10**m) for m in range(3) for k in (1, 2, 5))]
    parser.add_argument("--error_iter", type=int, default=100,
                        help="Run ROE checking every this many iterations, Default %(default)s iterations")
    parser.add_argument("-e", "--error_limit", type=float, default=0.4375,
                        help="Round off error (ROE) limit (0 - 0.47), Default %(default)s")
    parser.add_argument("--jacobi", action="store_false", default=True,
                        help="Do Jacobi Error Check (LL only), Default %(default)s")
    # Prime95/MPrime: 12 hours (JacobiErrorCheckingInterval)
    # GpuOwl v6.11: 500,000 iterations (must divide 10,000, usage says 1,000,000 iterations)
    parser.add_argument("--jacobi_iter", type=int, default=100000,
                        help="Run Jacobi Error Check every this many iterations (LL only), Default %(default)s iterations")
    parser.add_argument("--gerbicz", action="store_false", default=True,
                        help="Do Gerbicz Error Check (GEC) (PRP only), Default %(default)s")
    # Prime95/MPrime: 1,000, 1,000,000 iterations (PRPGerbiczCompareInterval),
    # Mlucas: 1,000, 1,000,000 iterations,
    # GpuOwl: 400 (-block <value>, must divide 10,000), 200,000 iterations (-log <step> value, must divide 10,000)
    parser.add_argument("--gerbicz_iter", type=int, default=100000,
                        help="Run GEC every this many iterations (PRP only), Default %(default)s iterations")
    parser.add_argument("--proof_files_dir", default=".",
                        help="Directory to hold large temporary proof files/residues, Default: %(default)s (current directory)")
    parser.add_argument(
        "--archive_proofs", help="Directory to archive PRP proof files after upload, Default: %(default)s")
    parser.add_argument("-u", "--username", default="ANONYMOUS",
                        help="GIMPS/PrimeNet User ID. Create a GIMPS/PrimeNet account: https://www.mersenne.org/update/. If you do not want a PrimeNet account, you can use ANONYMOUS.")
    parser.add_argument("-T", "--worktype", type=int, choices=[4, 100, 101, 102, 104, 150, 151, 152, 153, 154, 155, 160, 161], default=150, help="""Type of work, Default: %(default)s,
4 (P-1 factoring),
100 (smallest available first-time LL),
101 (double-check LL),
102 (world-record-sized first-time LL),
104 (100M digit number LL),
150 (smallest available first-time PRP),
151 (double-check PRP),
152 (world-record-sized first-time PRP),
153 (100M digit number PRP),
154 (smallest available first-time PRP that needs P-1 factoring),
155 (double-check using PRP with proof),
160 (first time Mersenne cofactors PRP),
161 (double-check Mersenne cofactors PRP)
"""
                        )
    parser.add_argument("--cert_work", action="store_false", default=True,
                        help="Get PRP proof certification work, Default: %default")
    parser.add_argument("--min_exp", type=int,
                        help="Minimum exponent to get from PrimeNet (2 - 999,999,999)")
    parser.add_argument("--max_exp", type=int,
                        help="Maximum exponent to get from PrimeNet (2 - 999,999,999)")
    parser.add_argument("-W", "--days_work", type=float, default=3.0,
                        help="Days of work to queue (1-90 days), Default: %(default)s days. Adds one to num_cache when the time left for the current assignment is less then this number of days.")
    parser.add_argument("-t", "--hours", type=float, default=6.0,
                        help="Hours between checkins and sending estimated completion dates, Default: %(default)s hours")
    parser.add_argument("-s", "--status", action="store_true", default=False,
                        help="Output a status report and any expected completion dates for all assignments and exit.")
    parser.add_argument("--upload_proofs", action="store_true", default=False,
                        help="Report assignment results, upload all PRP proofs and exit. Requires PrimeNet User ID.")
    parser.add_argument("--unreserve", type=int,
                        help="Unreserve assignment and exit. Use this only if you are sure you will not be finishing this exponent. Requires that the instance is registered with PrimeNet.")
    parser.add_argument("--unreserve_all", action="store_true", default=False,
                        help="Unreserve all assignments and exit. Quit GIMPS immediately. Requires that the instance is registered with PrimeNet.")
    parser.add_argument("--no_more_work", action="store_true", default=False,
                        help="Prevent script from getting new assignments and exit. Quit GIMPS after current work completes.")
    parser.add_argument("-H", "--hostname", default=platform.node(),
                        help="Optional computer name, Default: %(default)s")
    parser.add_argument("--hours_day", type=int, default=24,
                        help="Hours per day you expect to run TensorPrime (1 - 24), Default: %(default)s hours. Used to give better estimated completion dates.")
    parser.add_argument("--64-bit", action="store_true", default=False,
                        help="Enable 64 bit on Jax")

    # args is a dictionary in python types, in a
    # per-flag key-value mapping, which can be
    # accessed via, eg, flags["prime"], which will
    # return the integer passed in.
    args = parser.parse_args()

    # enable 64 bit support
    if getattr(args, "64_bit"):
        from jax.config import config as jax_config
        jax_config.update("jax_enable_x64", True)
        global jnp_precision
        jnp_precision = jnp.float64

    # Initialize logger specific to our runtime
    init_logger("tensorprime.log")

    p = args.prime
    if not p or not is_prime(p):
        parser.error("runtime requires a prime number for testing!")
    logging.info(f"Testing p={p}")

    # Used for save checkpointing. If a user's
    # program crashes they can resume from an
    # existing save file.
    resume = True
    preval = saveload.load(args.resume, p)
    if preval is None:
        resume = False

    # We choose the signal length by rounding down
    # the exponent to the nearest power of 2 and
    # then dividing by two twice. Experimentally
    # this will work for almost all known Mersenne
    # primes on the TPU out of the box. If a known-
    # working Mersenne prime throws a precision
    # error exception, double the FFT length and try
    # again.
    siglen = args.fft if args.fft else 1 << max(
        1, int(math.log2(p / (10 if getattr(args, "64_bit") else 2.5))))
    logging.info(f"Using FFT length {siglen}")

    logging.info("Starting TensorPrime")
    logging.info("Starting Probable Prime Test.")
    logging.debug("Initializing arrays")
    bit_array, power_bit_array, weight_array = initialize_constants(
        p, siglen)
    logging.debug(f"bit_array: {bit_array}")
    logging.debug(f"power_bit_array: {power_bit_array}")
    logging.debug(f"weight_array: {weight_array}")
    logging.debug("Array initialization complete")
    start_time = time.perf_counter()

    if resume:
        logging.info(f"Resuming at iteration {preval['iteration']}")
        s = prptest(p, siglen, bit_array, power_bit_array, weight_array,
                    start_pos=preval["iteration"], s=preval["signal"],
                    d=preval["d"], prev_d=preval["d_prev"])
    else:
        s = prptest(p, siglen, bit_array, power_bit_array, weight_array)

    end_time = time.perf_counter()
    logging.debug(s)
    n = (1 << p) - 1
    is_probable_prime = result_is_nine(s, power_bit_array, n)
    logging.info(
        f"{p} tested in {timedelta(seconds=end_time - start_time)}: {'probably prime!' if is_probable_prime else 'composite'}")

    # Clean save checkpoint files now that the
    # program has finished.
    if not is_probable_prime or not is_known_mersenne_prime(p):
        saveload.clean(p)


@partial(jit, static_argnums=2)
def fill_base_array(base_array, bit_array, signal_length):
    @jit
    def body_fn(i, vals):
        (base, base_array, bit_array, signal_length) = vals
        base_array = base_array.at[i].set(jnp.power(2, base))
        base += bit_array[i]
        return (base, base_array, bit_array, signal_length)

    (base, base_array, bit_array, signal_length) = lax.fori_loop(0, signal_length, body_fn,
                                                                 (0, base_array, bit_array, signal_length))
    return base_array


@partial(jit, static_argnums=(1, 2))
def fill_bit_array(bit_array, exponent, signal_length):
    @jit
    def body_fn(i, vals):
        (bit_array, exponent, signal_length) = vals
        bit_array = bit_array.at[i - 1].set(
            jnp.ceil((exponent * i) / signal_length) - jnp.ceil(exponent * (i - 1) / signal_length))
        return (bit_array, exponent, signal_length)

    (bit_array, exponent, signal_length) = lax.fori_loop(0, signal_length + 1, body_fn,
                                                         (bit_array, exponent, signal_length))
    return bit_array


@partial(jit, static_argnums=2)
def fill_power_bit_array(power_bit_array, bit_array, signal_length):
    @jit
    def body_fn(i, vals):
        (power_bit_array, bit_array) = vals
        power_bit_array = power_bit_array.at[i].set(jnp.power(2, bit_array[i]))
        return (power_bit_array, bit_array)

    (power_bit_array, bit_array) = lax.fori_loop(
        0, signal_length, body_fn, (power_bit_array, bit_array))
    return power_bit_array


@partial(jit, static_argnums=(1, 2))
def fill_weight_array(weight_array, exponent, signal_length):
    def body_fn(i, vals):
        (weight_array, exponent, signal_length) = vals
        weight_array = weight_array.at[i].set(
            jnp.power(2, (jnp.ceil(exponent * i / signal_length) - (exponent * i / signal_length))))
        return (weight_array, exponent, signal_length)

    (weight_array, exponent, signal_length) = lax.fori_loop(0, signal_length, body_fn,
                                                            (weight_array, exponent, signal_length))
    return weight_array


#  The constants here are used by IBDWT and are
#  precalculated when the program runs to increase
# program speed.
def initialize_constants(exponent, signal_length):
    # Each digit in the signal at index i can occupy
    # at most bit_array[i] bits.
    bit_array = jnp.zeros(signal_length, dtype=jnp_precision)
    bit_array = fill_bit_array(bit_array, exponent, signal_length)

    # The maximum possible value of each digit at
    # index i in the signal is power_bit_array[i]-1.
    power_bit_array = jnp.zeros(signal_length, dtype=jnp_precision)
    power_bit_array = fill_power_bit_array(
        power_bit_array, bit_array, signal_length)

    # The weight array is an array of fractional
    # powers of two as described in
    # "Discrete Weighted Transforms"
    weight_array = jnp.zeros(signal_length, dtype=jnp_precision)
    weight_array = fill_weight_array(weight_array, exponent, signal_length)

    return bit_array, power_bit_array, weight_array


# This is performed at the end of the PRP loop to
# finish the "partial carry" performed at each
# iteration. In this step, the overflow from each
# digit is carried along and returned as
# "carry_val"
@jit
def firstcarry(signal, power_bit_array):
    carry = jnp.int32(0)

    def body_fun(i, vals):
        (carry_val, signal, power_bit_array) = vals
        base = power_bit_array[i]
        val = jnp.add(signal[i], carry_val)
        signal = signal.at[i].set(jnp.mod(val, base))
        carry_val = jnp.floor_divide(val, base)
        return (carry_val, signal, power_bit_array)

    (carry_val, signal, power_bit_array) = lax.fori_loop(
        0, signal.shape[0], body_fun, (0, signal, power_bit_array))

    return carry_val, signal


# TODO: Remove the while loop and see if a single carry pass is sufficient
# (it appears to be this way in existing GIMPS programs)
#
# This step takes the "carry_val" returned by the
# firstcarry() function and redistributes it
# throughout the signal. This is what gives us the
# "free mod" in the IBDWT algorithm.
@jit
def secondcarry(carryval, signal, power_bit_array):
    def wloop_cond(vals):
        (carryval, signal, power_bit_array) = vals
        return carryval != 0

    def forloop_body(i, vals):
        (carryval, signal, power_bit_array) = vals
        base = power_bit_array[i]
        v = carryval + signal[i]
        signal = signal.at[i].set(jnp.mod(v, base))
        carryval = jnp.floor_divide(v, base)
        return (carryval, signal, power_bit_array)

    def wloop_body(vals):
        return lax.fori_loop(0, signal.shape[0], forloop_body, vals)

    (carryval, signal, power_bit_array) = lax.while_loop(
        wloop_cond, wloop_body, (carryval, signal, power_bit_array))
    return signal


# The "partial carry" performs enough of a carry
# to prevent overflow errors during the squaring
# step of the following IBDWT loop
@jit
def partial_carry(signal, power_bit_array):
    def forloop_body(i, vals):
        (signal, power_bit_array, carry_values) = vals
        signal = jnp.add(signal, carry_values)
        carry_values = jnp.floor_divide(signal, power_bit_array)
        signal = jnp.mod(signal, power_bit_array)
        carry_values = jnp.roll(carry_values, 1)
        return (signal, power_bit_array, carry_values)

    carry_values = jnp.zeros(signal.shape[0])
    (signal, power_bit_array, carry_values) = lax.fori_loop(
        0, 3, forloop_body, (signal, power_bit_array, carry_values))

    signal = jnp.add(signal, carry_values)
    return signal


@jit
def weighted_transform(signal_to_transform, weight_array):
    weighted_signal = jnp.multiply(signal_to_transform, weight_array)
    transformed_weighted_signal = jnp.fft.fft(weighted_signal)
    return transformed_weighted_signal


@jit
def inverse_weighted_transform(transformed_weighted_signal, weight_array):
    weighted_signal = jnp.real(jnp.fft.ifft(transformed_weighted_signal))
    signal = jnp.divide(weighted_signal, weight_array)
    return signal


# Converts the signal to the "balanced-digit"
# representation described by Crandall & Fagin
# in "Discrete Weighted Transforms".
@jit
def balance(signal, power_bit_array):
    def subtract_and_carry(vals):
        (signal, power_bit_array, index) = vals
        signal = signal.at[index].set(signal[index] - power_bit_array[index])
        return (signal, 1)

    def set_carry_to_zero(vals):
        (signal, power_bit_array, index) = vals
        return (signal, 0)

    def body_fn(i, vals):
        (signal, carry_val, power_bit_array) = vals
        signal = signal.at[i].set(signal[i] + carry_val)
        (signal, carry_val) = lax.cond((signal[i] >= power_bit_array[i] / 2), subtract_and_carry, set_carry_to_zero,
                                       (signal, power_bit_array, i))
        return (signal, carry_val, power_bit_array)

    (signal, carry_val, power_bit_array) = lax.fori_loop(
        0, signal.shape[0], body_fn, (signal, 0, power_bit_array))
    signal = signal.at[0].set(signal[0] + carry_val)
    return signal


# Squares a number (mod 2^prime_exponent - 1) as
# described in "Discrete Weighted Transforms".
# This is functionally identital to a call to
# `multmod_with_ibdwt` where signal1 and signal2
# are identical, with the added benefit of 1 fewer
# FFT runs.
@partial(jit, static_argnums=(1, 2,))
def squaremod_with_ibdwt(signal, prime_exponent,
                         signal_length, power_bit_array, weight_array):
    balanced_signal = balance(signal, power_bit_array)
    transformed_signal = weighted_transform(balanced_signal, weight_array)
    squared_transformed_signal = jnp.multiply(
        transformed_signal, transformed_signal)
    squared_signal = inverse_weighted_transform(
        squared_transformed_signal, weight_array)
    rounded_signal = jnp.round(squared_signal)
    roundoff = jnp.max(jnp.abs(jnp.subtract(squared_signal, rounded_signal)))
    parially_carried_signal = partial_carry(rounded_signal, power_bit_array)
    return parially_carried_signal, roundoff


# Multiplies two numbers
# (mod 2^prime_exponent - 1) as described in
# "Discrete Weighted Transforms".
@partial(jit, static_argnums=(2, 3,))
def multmod_with_ibdwt(signal1, signal2, prime_exponent,
                       signal_length, power_bit_array, weight_array):
    balanced_signal1 = balance(signal1, power_bit_array)
    balanced_signal2 = balance(signal2, power_bit_array)
    transformed_signal1 = weighted_transform(balanced_signal1, weight_array)
    transformed_signal2 = weighted_transform(balanced_signal2, weight_array)
    multiplied_transformed_signal = jnp.multiply(
        transformed_signal1, transformed_signal2)
    multiplied_signal = inverse_weighted_transform(
        multiplied_transformed_signal, weight_array)
    rounded_signal = jnp.round(multiplied_signal)
    roundoff = jnp.max(jnp.abs(jnp.subtract(
        multiplied_signal, rounded_signal)))
    carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
    fullycarried_signal = secondcarry(
        carryval, firstcarried_signal, power_bit_array)
    return fullycarried_signal, roundoff


# These globals are set to the values calculated
# by Gerbicz error checking. If GEC discovers an
# error has occurred, it will roll back to these
# saved values.
gec_s_saved = None
gec_i_saved = None
gec_d_saved = None


def rollback():
    if jnp.shape(gec_s_saved) is None:
        raise Exception(
            "Gerbicz error checking found an error but had nothing to rollback to. Exiting")
    if jnp.shape(gec_d_saved) is None:
        raise Exception(
            "Gerbicz error checking found an error but had nothing to rollback to. Exiting")
    if gec_i_saved is None:
        raise Exception(
            "Gerbicz error checking found an error but had nothing to rollback to. Exiting")
    return gec_i_saved, gec_s_saved, gec_d_saved


def update_gec_save(i, s, d):
    global gec_i_saved
    global gec_s_saved
    global gec_d_saved
    gec_i_saved = i
    gec_s_saved = s.copy()
    gec_d_saved = d.copy()


def prptest(exponent, siglen, bit_array, power_bit_array,
            weight_array, start_pos=0, s=None, d=None, prev_d=None):
    # Load settings values for this function
    GEC_enabled = config.getboolean("TensorPrime", "GECEnabled")
    GEC_iterations = int(config.get("TensorPrime", "GECIter"))

    # Uses counters to avoid modulo check
    save_i_count = save_iter = int(config.get("TensorPrime", "SaveIter"))
    print_i_count = print_iter = int(config.get("TensorPrime", "PrintIter"))
    if s is None:
        s = jnp.zeros(siglen).at[0].set(3)
    i = start_pos

    current_time = start = time.perf_counter_ns()
    while i < exponent:
        # Create a save checkpoint every save_i_count
        # iterations.
        if save_i_count == 0:
            logging.info(
                f"Saving progress (performed every {save_iter} iterations)...")
            saveload.save(exponent, siglen, s, i)
            save_i_count = save_iter
        save_i_count -= 1

        # Print a progress update every print_i_count
        # iterations
        if print_i_count == 0:
            temp = time.perf_counter_ns()
            delta_time = temp - current_time
            current_time = temp
            logging.info(
                f"Time elapsed at iteration {i}: {timedelta(microseconds=(current_time - start) // 1000)}, {(delta_time / 1000) / print_iter:.2f} µs/iter")
            print_i_count = print_iter
        print_i_count -= 1

        # Gerbicz error checking
        if GEC_enabled:
            L = int(math.sqrt(GEC_iterations))
            L_2 = L * L
            three_signal = jnp.zeros(siglen).at[0].set(3)
            if d is None:
                prev_d = d = three_signal
                update_gec_save(i, s, d)

            # Every L iterations, update d and prev_d
            if i != 0 and i % L == 0:
                prev_d = d
                d, roundoff = multmod_with_ibdwt(
                    d, s, exponent, siglen, power_bit_array, weight_array)
            # Every L^2 iterations, check the current d value with and independently calculated d
            if (i != 0 and i % L_2 == 0) or (i %
                                             L == 0 and (i + L > exponent)):
                prev_d_pow_signal = prev_d
                for j in range(L):
                    prev_d_pow_signal, roundoff = squaremod_with_ibdwt(prev_d_pow_signal, exponent, siglen,
                                                                       power_bit_array, weight_array)
                check_value, roundoff = multmod_with_ibdwt(three_signal, prev_d_pow_signal, exponent, siglen,
                                                           power_bit_array, weight_array)

                if not jnp.array_equal(d, check_value):
                    logging.error("error occurred. rolling back to last save.")
                    i, s, d = rollback()

                else:
                    logging.info("updating gec_save")
                    update_gec_save(i, s, d)

        # Running squaremod
        s, roundoff = squaremod_with_ibdwt(
            s, exponent, siglen, power_bit_array, weight_array)

        # Quick check to avoid roundoff errors. If a
        # roundoff error is encountered we have no
        # current method for dealing with it, so throw
        # an exception and terminate the program.
        if roundoff > 0.40625:
            logging.warning(f"Roundoff (iteration {i}): {roundoff}")
            if roundoff > 0.4375:
                raise Exception(
                    f"Roundoff error exceeded threshold (iteration {i}): {roundoff} vs 0.4375")

        i += 1

    # The "partial carry" may leave some values in
    # an incorrect state. Running a final carry
    # will clean this up to produce the residue we
    # want to check.
    carry_val, s = firstcarry(s, power_bit_array)
    s = secondcarry(carry_val, s, power_bit_array)
    return s


# Sum up the values in the signal until the total
# is 9. If there are any values left in the signal
# we know the total value cannot be 9.
def result_is_nine(signal, power_bit_array, n):
    signal = np.array(signal)  # copy signal array to CPU
    res = base = 0
    i = 0
    nine = 9 % n
    while res < nine and i < signal.shape[0]:
        res += int(signal[i]) * (1 << base)
        base += int(power_bit_array[i])
        i += 1
    return res == nine and not signal[i:].any()


# Kick off the main() function after defining
# all functions in the file
main()
