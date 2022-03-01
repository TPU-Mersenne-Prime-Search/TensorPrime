import argparse
import sys
import logging
import time

import numpy as np

from log_helper import init_logger

import jax
import jax.numpy as jnp
from jax import jit, lax, device_put
from functools import partial
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

# Global variables
import config

def main():
    print("Starting TensorPrime")
    parser = argparse.ArgumentParser()

    # in order to add more arguments to the parser, attempt a similar declaration to below. Anthing without a dash is becomes ordinal and required
    parser.add_argument("-p", "--prime", type=int,
                        help="seed for the mersenne prime being tested", default=None)
    parser.add_argument("--ll", action="store_true")
    parser.add_argument("--fft", type=str, default=None)
    parser.add_argument("--bench", action="store_true",
                        help="perform testing etc")
    parser.add_argument("--siglen", type=int, default=128,
                       help="Power of two used as the signal length")
    
    args = vars(parser.parse_args())
    if not args["prime"]:
        raise ValueError("runtime requires a prime number for testing!")
        exit()
    print(f"Testing p={args['prime']}")

    # args is a dictionary in python types, in a per flag key-value mapping, which can be accessed via,
    # eg, flags["prime"], which will return the integer passed in.
    # If you want specific behavior for the options, eg prime is none, exit()""
    # Command line arguments

    # Argument option to load from file?

    # Determine which function is wanted,
    # Run relevant function
    
    m_logger = init_logger(args)   # logging functionality specific to our runtime
    
    # Some may share global setups
    #if args["prime"] and args["siglen"]:
    # Pass prime power and signal length.
    config.initialize_constants(args["prime"], args["siglen"])

    if args["bench"] is not None:
        pass

    if args["fft"] is not None:
        pass
  
    if args["prime"] is not None:
        p = int(args["prime"])
        print("Starting Probable Prime Test.")
        print("Initializing arrays")
        bit_array, power_bit_array, weight_array = initialize_constants(p, config.signal_length)
        print(f"bit_array: {bit_array}")
        print(f"power_bit_array: {power_bit_array}")
        print(f"weight_array: {weight_array}")
        print("Array initialization complete")
        start_time = time.time()
        s = prptest(p, config.signal_length, bit_array, power_bit_array, weight_array)
        print(s)
        is_probable_prime = result_is_nine(s, bit_array, power_bit_array)
        end_time = time.time()
        print("{} tested in {} sec: {}".format(p, end_time - start_time,
                                               "probably prime!" if is_probable_prime else "composite"))

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

# @partial(jit, static_argnums=(0,1))
def prptest(exponent, siglen, bit_array, power_bit_array, weight_array):
  s = jnp.zeros(siglen).at[0].set(3)
  for i in range(exponent):
    if i%100 == 0:
      print(i)
    s = squaremod_with_ibdwt(s, exponent, siglen, power_bit_array, weight_array)
    #s = multmod_with_ibdwt(s, s, exponent, siglen, power_bit_array, weight_array)
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