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
GEC_enabled = True
GEC_iterations = 1000


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
    parser.add_argument("-r", "--resume", type=int, default = -1,
                        help="Save to resume from. Most recent is 0")
    
    args = vars(parser.parse_args())
    
    # Get values from memory
    # This WILL override the siglen given from arguments.
    if args["resume"] != -1 or config.settings["AutoResume"]:
        #args["resume"]
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
        
        s = None
        if args["resume"]:
            print("Resuming at iteration", args["iteration"])
            s = prptest(p, siglen, bit_array, power_bit_array, weight_array, startPos=args["iteration"], s=args["signal"])
        else:
            s = prptest(p, siglen, bit_array, power_bit_array, weight_array)
        
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
    bit_array = jnp.zeros(signal_length, dtype=jnp.float32)
    bit_array = fill_bit_array(bit_array, exponent, signal_length)
    
    power_bit_array = jnp.zeros(signal_length, dtype=jnp.float32)
    power_bit_array = fill_power_bit_array(power_bit_array, bit_array, signal_length)
    
    weight_array = jnp.zeros(signal_length, dtype=jnp.float32)
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
        (signal, carry_val) = lax.cond((signal[i] >= power_bit_array[i] / 2), subtract_and_carry, set_carry_to_zero, (signal, power_bit_array, i))
        return (signal, carry_val, power_bit_array)
    (signal, carry_val, power_bit_array) = lax.fori_loop(0, signal.shape[0], body_fn, (signal, 0, power_bit_array))
    signal = signal.at[0].set(signal[0] + carry_val)
    return signal

@partial(jit, static_argnums=(1,2,))
def squaremod_with_ibdwt(signal, prime_exponent, signal_length, power_bit_array, weight_array):
    balanced_signal = balance(signal, power_bit_array)
    transformed_signal = weighted_transform(balanced_signal, weight_array)
    squared_transformed_signal = jnp.multiply(transformed_signal, transformed_signal)
    squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array)
    rounded_signal = jnp.int32(jnp.round(squared_signal))

    roundoff = jnp.max(jnp.abs(jnp.subtract(squared_signal, rounded_signal)))

    # Balance the digits ( )
    carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
    fullycarried_signal = secondcarry(carryval, firstcarried_signal, power_bit_array)
    return fullycarried_signal, roundoff

@partial(jit, static_argnums=(2,3,))
def multmod_with_ibdwt(signal1, signal2, prime_exponent, signal_length, power_bit_array, weight_array):
    balanced_signal1 = balance(signal1, power_bit_array)
    balanced_signal2 = balance(signal2, power_bit_array)

    transformed_signal1 = weighted_transform(balanced_signal1, weight_array)
    transformed_signal2 = weighted_transform(balanced_signal2, weight_array)
    multiplied_transformed_signal = jnp.multiply(transformed_signal1, transformed_signal2)
    multiplied_signal = inverse_weighted_transform(multiplied_transformed_signal, weight_array)
    rounded_signal = jnp.int32(jnp.round(multiplied_signal))
    
    roundoff = jnp.max(jnp.abs(jnp.subtract(multiplied_signal, rounded_signal)))

    # Balance the digits ( )
    carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
    fullycarried_signal = secondcarry(carryval, firstcarried_signal, power_bit_array)
    return fullycarried_signal, roundoff

gec_s_saved = None
gec_i_saved = None
gec_d_saved = None

def rollback():
  if gec_s_saved == None:
    raise Exception("Gerbicz error checking found an error but had nothing to rollback to. Exiting")
  if gec_i_saved == None:
    raise Exception("Gerbicz error checking found an error but had nothing to rollback to. Exiting")
  if gec_d_saved == None:
    raise Exception("Gerbicz error checking found an error but had nothing to rollback to. Exiting")
  return gec_i_saved, gec_s_saved, gec_d_saved

def update_gec_save(i, s, d):
    global gec_i_saved
    global gec_s_saved
    global gec_d_saved
    gec_i_saved = i
    gec_s_saved = s.copy()
    gec_d_saved = d.copy()

def prptest(exponent, siglen, bit_array, power_bit_array, weight_array, startPos = 0, s = None):

  start = time.time()
  # Load settings values for this function
  timestamp = config.settings["Timestamps"]
  # Uses counters to avoid modulo check
  saveIter = config.settings["SaveIter"]
  saveIcount = saveIter
  printIter = config.settings["PrintIter"]
  printIcount = printIter
  if s == None:
    s = jnp.zeros(siglen).at[0].set(3)
  i = startPos

  if GEC_enabled:
    L = GEC_iterations
    L_2 = L*L
    three_signal = jnp.zeros(siglen).at[0].set(3)
    d = s.copy()
    prev_d = d
    update_gec_save(i, s, d)

  while(i < exponent):
  
    # Gerbicz error checking
    if GEC_enabled:
      # Every L iterations, update d and prev_d
      if i != 0 and i % L == 0:
        prev_d = d
        d, roundoff = multmod_with_ibdwt(d, s, exponent, siglen, power_bit_array, weight_array)
      # Every L^2 iterations, check the current d value with and independently calculated d
      if (i != 0 and i % L_2 == 0) or (i % L == 0 and (i + L > exponent)):
        prev_d_pow_signal = prev_d
        for j in range(L):
          prev_d_pow_signal, roundoff = squaremod_with_ibdwt(prev_d_pow_signal, exponent, siglen, power_bit_array, weight_array)
        check_value, roundoff = multmod_with_ibdwt(three_signal, prev_d_pow_signal, exponent, siglen, power_bit_array, weight_array)
        
        if not jnp.array_equal(d, check_value):
          print("error occurred. rolling back to last save.")
          i,s,d = rollback()
      
        else:
          print("updating gec_save")
          update_gec_save(i,s,d)
    
    # Saving
    if saveIcount == 0:
      saveload.save(exponent, siglen, s, i)
      saveIcount = saveIter
    saveIcount -= 1

    s, roundoff = squaremod_with_ibdwt(s, exponent, siglen, power_bit_array, weight_array)
    if roundoff > 0.4375:
      raise Exception(f"Roundoff error exceeded threshold (iteration {i}): {roundoff} vs 0.4375")
      
    # Printing
    if timestamp:
      if printIcount == 0:
        time_elapsed = time.time() - start
        print("Time elapsed at iteration ", i, ": ", time_elapsed, ". S = ", s)
        printIcount = printIter
      printIcount -= 1

    i += 1
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