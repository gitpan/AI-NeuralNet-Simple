package AI::NeuralNet::Simple;

$REVISION = '$Id: Simple.pm,v 1.1.1.1 2003/10/05 14:16:14 ovid Exp $';
$VERSION  = '0.01';

use strict;
use warnings;

sub new
{
    my ($class,@args) = @_;
    die "You must supply three positive integers to new()" unless 3 == @args;
    foreach (@args) {
        die "Arguments to new() must be positive integers" unless defined $_ && /^\d+/;
    }
    _new_network(@args);
    my $self = bless {
        input  => $args[0],
        hidden => $args[1],
        output => $args[2],
    }, $class;
    $self->iterations(10000); # set a reasonable default
}

sub train
{
    my ($self, $inputref, $outputref) = @_;
    _train($inputref, $outputref);
    $self;
}

sub train_set
{
    my ($self, $set, $iterations) = @_;
    $iterations ||= $self->iterations;
    _train_set($set,$iterations);
    $self;
}

sub iterations
{
    my ($self,$iterations) = @_;
    if (defined $iterations) {
        die "iterations() value must be a positive integer."
            unless $iterations and $iterations =~ /^\d+$/;
        $self->{iterations} = $iterations;
        return $self;
    }
    $self->{iterations};
}

sub infer
{
    my ($self,$data) = @_;
    _infer($data);
}

sub winner 
{
    # returns index of largest value in inferred answer
    my ($self,$data) = @_;
    my $arrayref     = _infer($data);

    my $largest      = 0;
    for (0 .. $#$arrayref) {
        $largest = $_ if $arrayref->[$_] > $arrayref->[$largest];
    }
    return $largest;
}

sub DESTROY
{
    _destroy_network();
}

use Inline C => <<'END_OF_C_CODE';

/*
 * Macros and symbolic constants
 */

#define RAND_WEIGHT ( ((float)rand() / (float)RAND_MAX) - 0.5 )

#define getSRand() ((float)rand() / (float)RAND_MAX)
#define getRand()  (int)((x) * getSRand())

#define sqr(x) ((x) * (x))

typedef struct {
    double **input_to_hidden;
    double **hidden_to_output;
} SYNAPSE;

SYNAPSE weight;

typedef struct {
    double *hidden;
    double *output;
} ERROR;

ERROR error;

typedef struct {
    double *input;
    double *hidden;
    double *output;
    double *target;
} LAYER;

LAYER neuron;

typedef struct {
    int input;
    int hidden;
    int output;
} NEURON_COUNT;

NEURON_COUNT size;

typedef struct {
    float        learn_rate;
    SYNAPSE      weight;
    ERROR        error;
    LAYER        neuron;
    NEURON_COUNT size;
} NEURAL_NETWORK;

NEURAL_NETWORK network;

AV*    get_array_from_aoa(SV* scalar, int index);
AV*    get_array(SV* aref);
SV*    get_element(AV* array, int index);

double sigmoid_derivative(double val);
double sigmoid(double val);
float  get_float_element(AV* array, int index);
int    is_array_ref(SV* ref);
void   _assign_random_weights(void);
void   _back_propogate(void);
void   _destroy_network();
void   _feed(double *input, double *output, int learn);
void   _feed_forward(void);

int is_array_ref(SV* ref)
{
    if (SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVAV)
        return 1;
    else
        return 0;
}

double sigmoid(double val)
{
    return (1.0 / (1.0 + exp(-val)));
}

double sigmoid_derivative(double val)
{
    return (val * (1.0 - val));
}

AV* get_array(SV* aref)
{
    if (! is_array_ref(aref))
        croak("get_array() argument is not an array reference");
    
    return (AV*)SvRV(aref);
}

float get_float_element(AV* array, int index)
{
    SV *elem = get_element(array, index);

    if (looks_like_number(elem))
        return atof(SvPV(elem, PL_na));
    else
        croak("Element did not look like a number");
}

SV* get_element(AV* array, int index)
{
    SV   **temp;
    temp = av_fetch(array, index, 0);
    if (!temp) {
        printf("Could not fetch element %d from array", index);
        croak("Ending program");
    }
    else
        return *temp;
}

AV* get_array_from_aoa(SV* aref, int index)
{
    SV *elem;
    AV *array;

    /* dereference array and get requested arrayref */
    array  = get_array(aref);
    elem   = get_element(array, index);
    
    /* dereference array ref */
    return get_array(elem);
}

/*
 * we'll come back later and clean these up!
 */

int _create_network(void)
{
    int i;
    /* each of the next two variables has an extra row for the "bias" */
    int input_layer_with_bias  = network.size.input  + 1;
    int hidden_layer_with_bias = network.size.hidden + 1;
    network.learn_rate = .2;

    network.neuron.input  = malloc(sizeof(double) * network.size.input);
    network.neuron.hidden = malloc(sizeof(double) * network.size.hidden);
    network.neuron.output = malloc(sizeof(double) * network.size.output);
    network.neuron.target = malloc(sizeof(double) * network.size.output);

    network.error.hidden  = malloc(sizeof(double) * network.size.hidden);
    network.error.output  = malloc(sizeof(double) * network.size.output);
    
    /* one extra for sentinel */
    network.weight.input_to_hidden  
        = malloc(sizeof(void *) * (input_layer_with_bias + 1));
    network.weight.hidden_to_output 
        = malloc(sizeof(void *) * (hidden_layer_with_bias + 1));

    if(!network.weight.input_to_hidden || !network.weight.hidden_to_output) {
        printf("Initial malloc() failed\n");
        return 0;
    }
    
    /* now allocate the actual rows */
    for(i = 0; i < input_layer_with_bias; i++) {
        network.weight.input_to_hidden[i] 
            = malloc(hidden_layer_with_bias * sizeof(double));
        if(network.weight.input_to_hidden[i] == 0) {
            free(*network.weight.input_to_hidden);
            printf("Second malloc() to weight.input_to_hidden failed\n");
            return 0;
        }
    }

    /* now allocate the actual rows */
    for(i = 0; i < hidden_layer_with_bias; i++) {
        network.weight.hidden_to_output[i] 
            = malloc(network.size.output * sizeof(double));
        if(network.weight.hidden_to_output[i] == 0) {
            free(*network.weight.hidden_to_output);
            printf("Second malloc() to weight.hidden_to_output failed\n");
            return 0;
        }
    }

    /* initialize the sentinel value */
    network.weight.input_to_hidden[input_layer_with_bias]   = 0;
    network.weight.hidden_to_output[hidden_layer_with_bias] = 0;

    return 1;
}

void _destroy_network()
{
    double **row;

    for(row = network.weight.input_to_hidden; *row != 0; row++) {
        free(*row);
    }
    free(network.weight.input_to_hidden);

    for(row = network.weight.hidden_to_output; *row != 0; row++) {
        free(*row);
    }
    free(network.weight.hidden_to_output);

    free(network.neuron.input);
    free(network.neuron.hidden);
    free(network.neuron.output);
    free(network.neuron.target);

    free(network.error.hidden);
    free(network.error.output);
}

/*
 * Support functions for back propogation
 */

void _assign_random_weights(void)
{
    int hid, inp, out;

    for (inp = 0; inp < network.size.input + 1; inp++) {
        for (hid = 0; hid < network.size.hidden; hid++) {
            network.weight.input_to_hidden[inp][hid] = RAND_WEIGHT;
        }
    }

    for (hid = 0; hid < network.size.hidden + 1; hid++) {
        for (out = 0; out < network.size.output; out++) {
            network.weight.hidden_to_output[hid][out] = RAND_WEIGHT;
        }
    }
}

/*
 * Feed-forward Algorithm
 */

void _feed_forward(void)
{
    int inp, hid, out;
    double sum;

    /* calculate input to hidden layer */
    for (hid = 0; hid < network.size.hidden; hid++) {

        sum = 0.0;
        for (inp = 0; inp < network.size.input; inp++) {
            sum += network.neuron.input[inp]
                * network.weight.input_to_hidden[inp][hid];
        }

        /* add in bias */
        sum += network.weight.input_to_hidden[network.size.input][hid];

        network.neuron.hidden[hid] = sigmoid(sum);
    }

    /* calculate the hidden to output layer */
    for (out = 0; out < network.size.output; out++) {

        sum = 0.0;
        for (hid = 0; hid < network.size.hidden; hid++) {
            sum += network.neuron.hidden[hid] 
                * network.weight.hidden_to_output[hid][out];
        }

        /* add in bias */
        sum += network.weight.hidden_to_output[network.size.hidden][out];

        network.neuron.output[out] = sigmoid(sum);
    }
}

/*
 * Backpropogation algorithm.  This is where the learning gets done.
 */

void _back_propogate(void)
{
    int inp, hid, out;

    /* calculate the output layer error (step 3 for output cell) */
    for (out = 0; out < network.size.output; out++) {
        network.error.output[out] 
            = (network.neuron.target[out] - network.neuron.output[out]) 
              * sigmoid_derivative(network.neuron.output[out]);
    }

    /* calculate the hidden layer error (step 3 for hidden cell) */
    for (hid = 0; hid < network.size.hidden; hid++) {

        network.error.hidden[hid] = 0.0;
        for (out = 0; out < network.size.output; out++) {
            network.error.hidden[hid] 
                += network.error.output[out] 
                 * network.weight.hidden_to_output[hid][out];
        }
        network.error.hidden[hid] 
            *= sigmoid_derivative(network.neuron.hidden[hid]);
    }

    /* update the weights for the output layer (step 4) */
    for (out = 0; out < network.size.output; out++) {
        for (hid = 0; hid < network.size.hidden; hid++) {
            network.weight.hidden_to_output[hid][out] 
                += (network.learn_rate 
                  * network.error.output[out] 
                  * network.neuron.hidden[hid]);
        }

        /* update the bias */
        network.weight.hidden_to_output[network.size.hidden][out] 
            += (network.learn_rate 
              * network.error.output[out]);
    }

    /* update the weights for the hidden layer (step 4) */
    for (hid = 0; hid < network.size.hidden; hid++) {

        for  (inp = 0; inp < network.size.input; inp++) {
            network.weight.input_to_hidden[inp][hid] 
                += (network.learn_rate 
                  * network.error.hidden[hid] 
                  * network.neuron.input[inp]);
        }

        /* update the bias */
        network.weight.input_to_hidden[network.size.input][hid] 
            += (network.learn_rate 
              * network.error.hidden[hid]);
    }
}

void _train(SV* input, SV* output)
{
    int i,length;
    AV *array;
    SV *elem;
    double *input_array  = malloc(sizeof(double) * network.size.input);
    double *output_array = malloc(sizeof(double) * network.size.output);

    if (! is_array_ref(input) || ! is_array_ref(output)) {
        croak("train() takes two arrayrefs.");
    }
    
    array  = get_array(input);
    length = av_len(array)+ 1;
    
    if (length != network.size.input) {
        croak("Length of input array does not match network");
    }
    for (i = 0; i < length; i++) {
        input_array[i] = get_float_element(array, i);
    }

    array  = get_array(output);
    length = av_len(array) + 1;
    
    if (length != network.size.output) {
        croak("Length of output array does not match network");
    }
    for (i = 0; i < length; i++) {
        output_array[i] = get_float_element(array, i);
    }

    _feed(input_array, output_array, 1);

    free(input_array);
    free(output_array);
}

int _new_network(int input, int hidden, int output)
{
    network.size.input  = input;
    network.size.hidden = hidden;
    network.size.output = output;

    if ( ! _create_network() ) {
        printf("Failure initializing synapse weights\n");
        return 0;
    }

    /* seed the random number generator */
    srand(time(NULL));
    _assign_random_weights();
    return 1;
}

void _train_set(SV* set, int iterations)
{
    AV     *input_array, *output_array; /* perl arrays */
    double *input, *output; /* C arrays */
    double error;

    int set_length=0, input_length=0, output_length=0;
    int i,j;
    int index;

    set_length = av_len(get_array(set))+1;
    
    if (!set_length)
        croak("_train_set() array ref has no data");
    if (set_length % 2)
        croak("_train_set array ref must have an even number of elements");

    /* allocate memory for out input and output arrays */
    input_array    = get_array_from_aoa(set, 0);
    input          = malloc(sizeof(double) * set_length * (av_len(input_array)+1));

    output_array    = get_array_from_aoa(set, 1);
    output          = malloc(sizeof(double) * set_length * (av_len(output_array)+1));

    for (i=0; i < set_length; i += 2) {
        input_array = get_array_from_aoa(set, i);
        
        if (av_len(input_array)+1 != network.size.input)
            croak("Length of input data does not match");
        
        /* iterate over the input_array and assign the floats to input */
        
        for (j = 0; j < network.size.input; j++) {
            index = (i/2*network.size.input)+j;
            input[index] = get_float_element(input_array, j); 
        }
        
        output_array = get_array_from_aoa(set, i+1);
        if (av_len(output_array)+1 != network.size.output)
            croak("Length of output data does not match");

        for (j = 0; j < network.size.output; j++) {
            index = (i/2*network.size.output)+j;
            output[index] = get_float_element(output_array, j); 
        }
    }

    for (i = 0; i < iterations; i++) {
        for (j = 0; j < (set_length/2); j++) {
            _feed(&input[j*network.size.input], &output[j*network.size.output], 1);
            if (! (i % 1000)) {
                error = 0.0;
                for (index = 0; index < network.size.output; index++) {
                    int out_index = index + (j * network.size.output);
                    error += sqr( (output[out_index] - network.neuron.output[index]) );
                }
                error = 0.5 * error;
                //printf("mse = %f\n", error);
            }
        }
    }
    free(input);
    free(output);
}

SV* _infer(SV *array_ref)
{
    int    i;
    double *c_array, *dummy;
    AV     *perl_array, *result = newAV();

    /* feed the data */
    perl_array = get_array(array_ref);
    c_array    = malloc(sizeof(double) * network.size.input);

    for (i = 0; i < network.size.output; i++)
        c_array[i] = get_float_element(perl_array, i);

    _feed(c_array, dummy, 0); 
    free(c_array);

    /* read the results */
    for (i = 0; i < network.size.output; i++) {
        av_push(result, newSVnv(network.neuron.output[i]));
    }
    return newRV_noinc((SV*) result);
}

void _feed(double *input, double *output, int learn)
{
    int i;

    for (i=0; i < network.size.input; i++) {
        network.neuron.input[i]  = input[i];
    }

    if (learn)
        for (i=0; i < network.size.output; i++)
            network.neuron.target[i] = output[i];

    _feed_forward();

    if (learn) _back_propogate(); 
}

/*
 *  The original author of this code is M. Tim Jones <mtj@cogitollc.com> and
 *  written for the book "AI Application Programming", by Charles River Media.
 *
 *  It's been so heavily modified but credit should be given where credit is
 *  due.  Therefore ...
 *
 *  Copyright (c) 2003 Charles River Media.  All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, is hereby granted without fee provided that the following
 *  conditions are met:
 * 
 *    1.  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.  2.
 *    Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.  3.  Neither the
 *    name of Charles River Media nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY CHARLES RIVER MEDIA AND CONTRIBUTORS 'AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL CHARLES RIVER MEDIA OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
END_OF_C_CODE


1;

__END__

=head1 NAME

AI::NeuralNet::Simple - A simple learning module for building neural nets.

=head1 SYNOPSIS

  use AI::NeuralNet::Simple;
  my $net = AI::NeuralNet::Simple->new(2,1,2);
  # teach it logical 'or'
  for (1 .. 10000) {
      $net->train([1,1],[0,1]);
      $net->train([1,0],[0,1]);
      $net->train([0,1],[0,1]);
      $net->train([0,0],[1,0]);
  }
  printf "Answer: %d\n",   $net->winner([1,1]);
  printf "Answer: %d\n",   $net->winner([1,0]);
  printf "Answer: %d\n",   $net->winner([0,1]);
  printf "Answer: %d\n\n", $net->winner([0,0]);

=head1 ABSTRACT

  This module is a simple neural net learning tool designed for those who have
  an interest in artificial intelligence but need a "gentle" introduction.
  This is not intended to replace any of the neural net modules currently
  available on the CPAN.  Instead, the documentation is designed to be easy
  enough for a beginner to understand.

=head1 DESCRIPTION

=head2 The Disclaimer

Please note that the following information is terribly incomplete.  That's
deliberate.  Anyone familiar with neural networks is going to laugh themselves
silly at how simplistic the following information is and the astute reader will
notice that I've raised far more questions than I've answered.

So why am I doing this?  Because I'm giving I<just enough> information for
someone new to neural networks to have enough of an idea of what's going on so
they can actually use this module and then move on to something more powerful,
if interested.

=head2 The Biology

A neural network, at its simplest, is merely an attempt to mimic nature's
"design" of a brain.  Like many successful ventures in the field of artificial
intelligence, we find that blatantly ripping off natural designs has allowed us
to solve many problems that otherwise might prove intractable.  Fortunately,
Mother Nature has not chosen to apply for patents.

Our brains are comprised of neurons connected to one another by axons.  The
axon makes the actual connection to a neuron via a synapse.  When neurons
receive information, they process it and feed this information to other neurons
who in turn process the information and send it further until eventually
commands are sent to various parts of the body and muscles twitch, emotions are
felt and we start eyeing our neighbor's popcorn in the movie theater, wondering
if they'll notice if we snatch some while they're watching the movie.

=head2 A simple example of a neuron

Now that you have a solid biology background (uh, no), how does this work when
we're trying to simulate a neural network?  The simplest part of the network is
the neuron (also known as a node or, sometimes, a neurode).  A we might think
of a neuron as follows (OK, so I won't make a living as an ASCII artist):

Input neurons   Synapses   Neuron   Output

                            ----
  n1            ---w1----> /    \
  n2            ---w2---->|  n4  |---w4---->
  n3            ---w3----> \    /
                            ----

(Note that the above doesn't quite match what's in the C code for this module,
but it's close enough for you to get the idea.  This is one of the many
oversimplifications that have been made).

In the above example, we have three input neurons (n1, n2, and n3).  These
neurons feed whatever output they have through the three synapses (w1, w2, w3)
to the neuron in question, n4.  The three synapses each have a "weight", which
is an amount that the input neurons' output is multiplied by.  

The neuron n4 computes its output with something similar to the following:

  output = 0

  foreach (input.neuron)
      output += input.neuron.output * input.neuron.synapse.weight

  ouput = activation_function(output)

The "activation function" is a special function that is applied to the inputs
to generate the actual output.  There are a variety of activation functions
available with three of the most common being the linear, sigmoid, and tahn
activation functions.  For technical reasons, the linear activation function
cannot be used with the type of network that C<AI::NeuralNet::Simple> employs.
This module uses the sigmoid activation function.  (More information about
these can be found by reading the information in the L<SEE ALSO> section or by
just searching with Google.)

Once the activation function is applied, the output is then sent through the
next synapse, where it will be multiplied by w4 and the process will continue.

=head2 C<AI::NeuralNet::Simple> architecture

The architecture used by this module has (at present) 3 fixed layers of
neurons: an input, hidden, and output layer.  In practice, a 3 layer network is
applicable to many problems for which a neural network is appropriate, but this
is not always the case.  In this module, we've settled on a fixed 3 layer
network for simplicity.

Here's how a three layer network might learn "logical or".  First, we need to
determine how many inputs and outputs we'll have.  The inputs are simple, we'll
choose two inputs as this is the minimum necessary to teach a network this
concept.  For the outputs, we'll also choose two neurons, with the neuron with
the highest output value being the "true" or "false" response that we are
looking for.  We'll only have one neuron for the input layer.  Thus, we get a
network that resembles the following:

         Input  Hidden  Output

 input1  ---->n1\    /---->n5---> output1
                 \  /
                  n3
                 /  \
 input2  ---->n2/    \---->n5---> output2

Let's say that output 1 will correspond to "false" and output 2 will correspond
to true.  If we feed 1 (or true) or both input 1 and input 2, we hope that output
2 will be true and output 1 will be false.  The following table should illustrate
the expected results:

 input   output
 1   2   1    2
 -----   ------
 1   1   0    1
 1   0   0    1
 0   1   0    1
 0   0   0    0

The type of network we use is a forward-feed back error propagation network,
referred to as a backpropagation network, for short.  The way it works is
simple.  When we feed in our input, it travels from the input to hidden layers
and then to the output layers.  This is the "feed forward" part.  We then
compare the output to the expected results and measure how far off we are.  We
then adjust the weights on the "output to hidden" synapses, measure the error
on the hidden nodes and then adjust the weights on the "hidden to input"
synapses.  This is what is referred to as "back error propagation".

We continue this process until the amount of error is small enough that we are
satisfied.  In reality, we will rarely if ever get precise results from the
network, but we learn various strategies to interpret the results.  In the
example above, we use a "winner takes all" strategy.  Which ever of the output
nodes has the greatest value will be the "winner", and thus the answer.

In the examples directory, you will find a program named "logical_or.pl" which
demonstrates the above process.

=head2 Building a network

In creating a new neural network, there are three basic steps:

=over 4

=item 1 Designing

This is choosing the number of layers and the number of neurons per layer.  In
C<AI::NeuralNet::Simple>, the number of layers is fixed.

With more complete neural net packages, you can also pick which activation
functions you wish to use and the "learn rate" of the neurons.

=item 2 Training

This involves feeding the neural network enough data until the error rate is
low enough to be acceptable.  Often we have a large data set and merely keep
iterating until the desired error rate is achieved.

=item 3 Measuring results

One frequent mistake made with neural networks is failing to test the network
with different data from the training data.  It's quite possible for a
backpropagation network to hit what is known as a "local minimum" which is not
truly where it should be.  This will cause false results.  To check for this,
after training we often feed in other known good data for verification.  If the
results are not satisfactory, perhaps a different number of neurons per layer
should be tried or a different set of training data should be supplied.

=back

=head1 Programming C<AI::NeuralNet::Simple>

=head2 C<new($input, $hidden, $output)>

C<new()> accepts three integers.  These number represent the number of nodes in
the input, hidden, and output layers, respectively.  To create the "logical or"
network described earlier:

  my $net = AI::NeuralNet::Simple->new(2,1,2);

=head2 C<train(\@input, \@output)>

This method trains the network to associate the input data set with the output
data set.  Representing the "logical or" is as follows:

  $net->train([1,1], [0,1]);
  $net->train([1,0], [0,1]);
  $net->train([0,1], [0,1]);
  $net->train([0,0], [1,0]);

Note that a one pass through the data is seldom sufficient to train a network.
In the example "logical or" program, we actually run this data through the
network ten thousand times.

  for (1 .. 10000) {
    $net->train([1,1], [0,1]);
    $net->train([1,0], [0,1]);
    $net->train([0,1], [0,1]);
    $net->train([0,0], [1,0]);
  }

=head2 C<train_set(\@dataset, [$iterations])>

Similar to train, this method allows us to train an entire data set at once.
It is typically faster than calling individual "train" methods.  The first
argument is expected to be an array ref of pairs of input and output array
refs.  The second argument is the number of iterations to train the set.  If
this argument is not provided here, you may use the C<iterations()> method to
set it (prior to calling C<train_set()>, of course).  A default of 10,000 will
be provided if not set.

  $net->train_set([
    [1,1], [0,1],
    [1,0], [0,1],
    [0,1], [0,1],
    [0,0], [1,0],
  ], 10000);

=head2 C<iterations([$integer])>

If called with a positive integer argument, this method will allow you to set
number of iterations that train_set will use.  If called without an argument,
it will return the number of iterations it was set to.

  $net->iterations(100000); # let's have lots more iterations!
  $net->iterations;         # returns 100000
  my @training_data = ( 
    [1,1], [0,1],
    [1,0], [0,1],
    [0,1], [0,1],
    [0,0], [1,0],
  );
  $net->train_set(\@training_data);
  
=head2 C<infer(\@input)>

This method, if provided with an input array reference, will return an array
reference corresponding to the output values that it is guessing.  Note that
these values will generally be close, but not exact.  For example, with the 
"logical or" program, you might expect results similar to:

  use Data::Dumper;
  print Dumper $net->infer([1,1]);
  
  $VAR1 = [
          '0.00993729281477686',
          '0.990100297418451'
        ];

That clearly has the second output item being close to 1, so as a helper method
for use with a winner take all strategy, we have ...

=head2 C<winner(\@input)>

This method returns the index of the highest value from inferred results:

  print $net->winner([1,1]); # will likely print "1"

For a more comprehensive example of how this is used, see the 
"examples/game_ai.pl" program.

=head1 EXPORT

None by default.

=head1 CAVEATS

This is B<alpha> code.  Very alpha.  Not even close to ready for production,
don't even think about it.  I'm putting it on the CPAN lest it languish on my
hard-drive forever.  Hopefully someone will get some use out of it and think to
send me a patch or two.

=head1 TODO

=over 4

=item * Allow user to set training rate

=item * Make MSE (mean squared error) public

=item * Save and restore networks

=item * Allow more than one network at a time

=item * Allow different activation functions

=item * Allow different numbers of layers

=back

=head1 BUGS

Probably.

=head1 SEE ALSO

"AI Application Programming by M. Tim Jones, copyright (c) by Charles River
Media, Inc.  

The C code in this module is based heavily upon Mr. Jones backpropogation
network in the book.  The "game ai" example in the examples directory is based
upon an example he has graciously allowed me to use.  I *had* to use it because
it's more fun than many of the dry examples out there :)

"Naturally Intelligent Systems", by Maureen Caudill and Charles Butler,
copyright (c) 1990 by Massachussetts Institute of Technology.

This book is a decent introduction to neural networks in general.  The forward
feed back error propogation is but one of many types.

=head1 AUTHOR

Curtis "Ovid" Poe, E<lt>eop_divo_sitruc@yahoo.comE<gt>

To email me, reverse "eop_divo_sitruc" in the email above.

=head1 COPYRIGHT AND LICENSE

Copyright 2003 by Curtis "Ovid" Poe

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself. 

=cut
