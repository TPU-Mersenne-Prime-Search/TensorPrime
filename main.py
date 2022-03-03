import argparse
import sys
import logging
import time
import math

import numpy as np

from log_helper import init_logger

import jax
import jax.numpy as jnp
from jax import jit, lax, device_put
from functools import partial
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

import config
import saveload

# Global variables
GEC_enabled = False
GEC_iterations = 2000000


def main():
    print("Starting TensorPrime")
    parser = argparse.ArgumentParser()
    
    config.getSettings()

    # in order to add more arguments to the parser, attempt a similar declaration to below. Anthing without a dash is becomes ordinal and required
    parser.add_argument("-p", "--prime", type=int,
                        help="seed for the mersenne prime being tested", default=None)
    parser.add_argument("--ll", action="store_true")
    parser.add_argument("--fft", type=str, default=None)
    parser.add_argument("--bench", action="store_true",
                        help="perform testing etc")
    parser.add_argument("--siglen", type=int, default=128,
                       help="Power of two used as the signal length")
    parser.add_argument("-r", "--resume", action="store_true")
    
    args = vars(parser.parse_args())
    
    # Get values from memory
    # This WILL override the siglen given from arguments.
    if args["resume"] or config.settings["AutoResume"]:
        preval = saveload.load()
        if preval != None:
            args.update(preval)
        else:
            args["resume"] = False
    else:
        args["resume"] = False
        
    if not args["prime"]:
        raise ValueError("runtime requires a prime number for testing!")
        exit()
    print(f"Testing p={args['prime']}")

    # args is a dictionary in python types, in a per flag key-value mapping, which can be accessed via,
    # eg, flags["prime"], which will return the integer passed in.
    # If you want specific behavior for the options, eg prime is none, exit()""
    # Command line arguments

    # Initialize logger specific to our runtime
    m_logger = init_logger(args)
    
    p = int(args["prime"])
    siglen = int(args["siglen"])

    if args["bench"] is not None:
        pass

    if args["fft"] is not None:
        pass
  
    if p is not None:
        print("Starting Probable Prime Test.")
        print("Initializing arrays")
        bit_array, power_bit_array, weight_array = initialize_constants(p, siglen)
        print(f"bit_array: {bit_array}")
        print(f"power_bit_array: {power_bit_array}")
        print(f"weight_array: {weight_array}")
        print("Array initialization complete")
        start_time = time.time()
        s = prptest(p, siglen, bit_array, power_bit_array, weight_array)
        '''
        start_time = time.time()
        is_probable_prime = None
        # Resume
        if args["resume"]:
            print("Resuming at iteration", args["iteration"])
            is_probable_prime = probable_prime(p, startPos=args["iteration"], s=args["signal"])
        else:
            is_probable_prime = probable_prime(p)
        '''
        end_time = time.time()
        print(s)
        is_probable_prime = result_is_nine(s, bit_array, power_bit_array)
        print("{} tested in {} sec: {}".format(p, end_time - start_time,
                                               "probably prime!" if is_probable_prime else "composite"))
    else:
      print("Usage: python -m main.py -p <exponent> [--siglen <signal length>]")

@partial(jit, static_argnums=2)
def fill_base_array(base_array, bit_array, signal_length):
  @jit
  def body_fn(i, vals):
    (base, base_array, bit_array, signal_length) = vals
    base_array = base_array.at[i].set(jnp.power(2, base))
    base += bit_array[i]
    return (base, base_array, bit_array, signal_length)
  (base, base_array, bit_array, signal_length) = lax.fori_loop(0, signal_length, body_fn, (0, base_array, bit_array, signal_length))
  return base_array

@partial(jit, static_argnums=(1,2))
def fill_bit_array(bit_array, exponent, signal_length):
  @jit
  def body_fn(i, vals):
    (bit_array, exponent, signal_length) = vals
    bit_array = bit_array.at[i - 1].set(
          jnp.ceil((exponent * i) / signal_length) - jnp.ceil(exponent * (i - 1) / signal_length))
    return (bit_array, exponent, signal_length)
  (bit_array, exponent, signal_length) = lax.fori_loop(0, signal_length+1, body_fn, (bit_array, exponent, signal_length))
  return bit_array

@partial(jit, static_argnums=2)
def fill_power_bit_array(power_bit_array, bit_array, signal_length):
  @jit
  def body_fn(i, vals):
    (power_bit_array, bit_array) = vals
    power_bit_array = power_bit_array.at[i].set(jnp.power(2, bit_array[i]))
    return (power_bit_array, bit_array)
  (power_bit_array, bit_array) = lax.fori_loop(0, signal_length, body_fn, (power_bit_array, bit_array))
  return power_bit_array

@partial(jit, static_argnums=(1,2))
def fill_weight_array(weight_array, exponent, signal_length):
  def body_fn(i, vals):
    (weight_array, exponent, signal_length) = vals
    weight_array = weight_array.at[i].set(
        jnp.power(2, (jnp.ceil(exponent * i / signal_length) - (exponent * i / signal_length))))
    return (weight_array, exponent, signal_length)
  (weight_array, exponent, signal_length) = lax.fori_loop(0, signal_length, body_fn, (weight_array, exponent, signal_length))
  return weight_array

@jit
def calc_max_array(power_bit_array):
  return jnp.subtract(power_bit_array, 1)


def initialize_constants(exponent, signal_length):
    bit_array = jnp.zeros(signal_length)
    bit_array = fill_bit_array(bit_array, exponent, signal_length)
    
    power_bit_array = jnp.zeros(signal_length)
    power_bit_array = fill_power_bit_array(power_bit_array, bit_array, signal_length)
    
    weight_array = jnp.zeros(signal_length)
    weight_array = fill_weight_array(weight_array, exponent, signal_length)

    return bit_array, power_bit_array, weight_array

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
  (carry_val, signal, power_bit_array) = lax.fori_loop(0, signal.shape[0], body_fun, (0, signal, power_bit_array))
  
  return carry_val, signal

# TODO: Remove the while loop and see if a single carry pass is sufficient
# (it appears to be this way in existing GIMPS programs)
@jit
def secondcarry(carryval, signal, power_bit_array):
  num_iter = 1
  def wloop_cond(vals):
    (carryval, signal, power_bit_array) = vals
    return carryval > 0
  
  def forloop_body(i, vals):
    (carryval, signal, power_bit_array) = vals
    base = power_bit_array[i]
    v = carryval + signal[i]
    signal = signal.at[i].set(jnp.mod(v, base))
    carryval = jnp.floor_divide(v, base)
    return (carryval, signal, power_bit_array)
  
  def wloop_body(vals):
    return lax.fori_loop(0, signal.shape[0], forloop_body, vals)
  
  (carryval, signal, power_bit_array) = lax.while_loop(wloop_cond, wloop_body, (carryval, signal, power_bit_array))
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

@partial(jit, static_argnums=(1,2,))
def squaremod_with_ibdwt(signal, prime_exponent, signal_length, power_bit_array, weight_array):
    transformed_signal = weighted_transform(signal, weight_array)
    squared_transformed_signal = jnp.multiply(transformed_signal, transformed_signal)
    squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array)
    # TODO: difference between pre-rounded and rounded signal for catching roundoff errors
    rounded_signal = jnp.int32(jnp.round(squared_signal))

    # Balance the digits ( )
    carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
    fullycarried_signal = secondcarry(carryval, firstcarried_signal, power_bit_array)
    return fullycarried_signal

@partial(jit, static_argnums=(2,3,))
def multmod_with_ibdwt(signal1, signal2, prime_exponent, signal_length, power_bit_array, weight_array):
    transformed_signal1 = weighted_transform(signal1, weight_array)
    transformed_signal2 = weighted_transform(signal2, weight_array)
    multiplied_transformed_signal = jnp.multiply(transformed_signal1, transformed_signal2)
    multiplied_signal = inverse_weighted_transform(multiplied_transformed_signal, weight_array)
    # TODO: difference between pre-rounded and rounded signal for catching roundoff errors
    rounded_signal = jnp.int32(jnp.round(multiplied_signal))

    # Balance the digits ( )
    carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
    fullycarried_signal = secondcarry(carryval, firstcarried_signal, power_bit_array)
    return fullycarried_signal


gec_s_saved = None
gec_i_saved = None

def rollback():
  if gec_s_saved == None or gec_i_saved == None:
    raise Exception("Gerbicz error checking found an error but had nothing to rollback to. Exiting")
  return gec_i_saved, gec_s_saved

def update_gec_save(i, s):
    gec_i_saved = i
    gec_s_saved = s

def prptest(exponent, siglen, bit_array, power_bit_array, weight_array):
  if GEC_enabled:
      print("setting GEC variables")
      L = GEC_iterations
      L_2 = L*L
      d = jnp.zeros(siglen).at[0].set(3)
      prev_d = jnp.zeros(siglen).at[0].set(3)
      print("GEC variables initialized")

  s = jnp.zeros(siglen).at[0].set(3)
  for i in range(exponent):
    
    # Print i every 100 iterations to track progress
    if i%100 == 0:
      print(i)
    
    # Gerbicz error checking
    if GEC_enabled:
      print("performing GEC checks")
      # Every L iterations, update d and prev_d
      if i != 0 and i % L == 0:
        print("updating d, s")
        prev_d = d
        d = multmod_with_ibdwt(d, s, exponent, siglen, power_bit_array, weight_array)
        # d = (d * s) % n
      # Every L^2 iterations, check the current d value with and independently calculated d
      if (i != 0 and i % L_2 == 0) or (i + L > exponent):
        print("checking value")
        three_signal = jnp.zeros(siglen).at[0].set(3)
        prev_d_pow_signal = None
        # Here: signalize (prev_d ** (2 ** L)) and store in prev_d_pow_signal
        for i in range(L):
          prev_d_pow_signal = squaremod_with_ibdwt(prev_d, siglen, power_bit_array, weight_array)
        check_value = multmod_with_ibdwt(three_signal, prev_d_pow_signal, siglen, power_bit_array, weight_array)
        # check_value = (3 * (prev_d ** (2 ** L))) % n
        if not jnp.equal(d, check_value):
          print("Error occured. Rolling back.")
          i,s = rollback()
        else:
          print("updating gec_save")
          update_gec_save(i,s)

    s = multmod_with_ibdwt(s, s, exponent, siglen, power_bit_array, weight_array)
  return s

def result_is_nine(signal, bit_array, power_bit_array):
  signal = np.array(signal) # copy signal array to CPU
  res = 0
  base = 0
  i = 0
  while res < 9 and i < signal.shape[0]:
    res += int(signal[i]) * (2**base)
    base += int(power_bit_array[i])
    i += 1
  return (res == 9 ) and (not signal[i:].any())

main()
