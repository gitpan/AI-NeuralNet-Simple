package AI::NeuralNet::Simple;

$REVISION = '$Id: Simple.pm,v 1.3 2004/01/31 20:34:11 ovid Exp $';
$VERSION  = '0.10';

use Log::Agent;

use strict;
use warnings;

sub handle	{ $_[0]->{handle} }

sub new
{
    my ($class, @args) = @_;
    logdie "you must supply three positive integers to new()"
		unless 3 == @args;
    foreach (@args) {
        logdie "arguments to new() must be positive integers"
			unless defined $_ && /^\d+/;
    }
	my $seed = rand(1);		# Perl invokes srand() on first call to rand()
    my $handle = c_new_network(@args);
	logdie "could not create new network" unless $handle >= 0;
    my $self = bless {
        input  => $args[0],
        hidden => $args[1],
        output => $args[2],
        handle => $handle,
    }, $class;
    $self->iterations(10000); # set a reasonable default
}

sub train
{
    my ($self, $inputref, $outputref) = @_;
    return c_train($self->handle, $inputref, $outputref);
}

sub train_set
{
    my ($self, $set, $iterations, $mse) = @_;
    $iterations ||= $self->iterations;
	$mse = -1.0 unless defined $mse;
    return c_train_set($self->handle, $set, $iterations, $mse);
}

sub iterations
{
    my ($self, $iterations) = @_;
    if (defined $iterations) {
        logdie "iterations() value must be a positive integer."
            unless $iterations and $iterations =~ /^\d+$/;
        $self->{iterations} = $iterations;
        return $self;
    }
    $self->{iterations};
}

sub delta
{
    my ($self, $delta) = @_;
	return c_get_delta($self->handle) unless defined $delta;
	logdie "delta() value must be a positive number" unless $delta > 0.0;
	c_set_delta($self->handle, $delta);
	return $self;
}

sub use_bipolar
{
    my ($self, $bipolar) = @_;
	return c_get_use_bipolar($self->handle) unless defined $bipolar;
	c_set_use_bipolar($self->handle, $bipolar);
	return $self;
}

sub infer
{
    my ($self,$data) = @_;
    c_infer($self->handle, $data);
}

sub winner 
{
    # returns index of largest value in inferred answer
    my ($self, $data) = @_;
    my $arrayref     = c_infer($self->handle, $data);

    my $largest      = 0;
    for (0 .. $#$arrayref) {
        $largest = $_ if $arrayref->[$_] > $arrayref->[$largest];
    }
    return $largest;
}

sub learn_rate
{
    my ($self, $rate) = @_;
    return c_get_learn_rate($self->handle) unless defined $rate; 
    logdie "learn rate must be between 0 and 1, exclusive"
		unless $rate > 0 && $rate < 1;
    c_set_learn_rate($self->handle, $rate);
    return $self;
}

sub DESTROY
{
	my $self = shift;
    c_destroy_network($self->handle);
}

#
# Serializing hook for Storable
#
sub STORABLE_freeze {
	my ($self, $cloning) = @_;
	my $internal = c_export_network($self->handle);

	# This is an excellent example where "we know better" than
	# the recommended way in Storable's man page...
	# Behaviour is the same whether we're cloning or not --RAM

	my %copy = %$self;
	delete $copy{handle};

	return("", \%copy, $internal);
}

#
# Deserializing hook for Storable
#
sub STORABLE_thaw {
	my ($self, $cloning, $x, $copy, $internal) = @_;
	%$self = %$copy;
	$self->{handle} = c_import_network($internal);
}

use Inline C => <<'END_OF_C_CODE';

/*
 * Macros and symbolic constants
 */

#define RAND_WEIGHT ( ((float)rand() / (float)RAND_MAX) - 0.5 )

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

typedef struct {
    float        learn_rate;
    double       delta;
    int          use_bipolar;
    SYNAPSE      weight;
    ERROR        error;
    LAYER        neuron;
    NEURON_COUNT size;
	double       *tmp;
} NEURAL_NETWORK;

int networks = 0;
NEURAL_NETWORK **network = NULL;

AV*    get_array_from_aoa(SV* scalar, int index);
AV*    get_array(SV* aref);
SV*    get_element(AV* array, int index);

double sigmoid(NEURAL_NETWORK *n, double val);
double sigmoid_derivative(NEURAL_NETWORK *n, double val);
float  get_float_element(AV* array, int index);
int    is_array_ref(SV* ref);
void   c_assign_random_weights(NEURAL_NETWORK *);
void   c_back_propagate(NEURAL_NETWORK *);
void   c_destroy_network(int);
void   c_feed(NEURAL_NETWORK *, double *input, double *output, int learn);
void   c_feed_forward(NEURAL_NETWORK *);
float  c_get_learn_rate(int);
void   c_set_learn_rate(int, float);
SV*    c_export_network(int handle);
int    c_import_network(SV *);

#define ABS(x)		((x) > 0.0 ? (x) : -(x))

int is_array_ref(SV* ref)
{
    if (SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVAV)
        return 1;
    else
        return 0;
}

double sigmoid(NEURAL_NETWORK *n, double val)
{
    return 1.0 / (1.0 + exp(-n->delta * val));
}

double sigmoid_derivative(NEURAL_NETWORK *n, double val)
{
	/*
	 * It's always called with val=sigmoid(x) and we want sigmoid'(x).
	 *
	 * Since sigmoid'(x) = delta * sigmoid(x) * (1 - sigmoid(x))
	 * the value we return is extremely simple.
	 *
	 * sigmoid_derivative(x) is NOT sigmoid'(x).
	 */

    return n->delta * val * (1.0 - val);
}

/* Not using tanh() as this is already defined in math headers */
double hyperbolic_tan(NEURAL_NETWORK *n, double val)
{
	double epx = exp(n->delta * val);
	double emx = exp(-n->delta * val);

	return (epx - emx) / (epx + emx);
}

double hyperbolic_tan_derivative(NEURAL_NETWORK *n, double val)
{
	/*
	 * It's always called with val=tanh(delta*x) and we want tanh'(delta*x).
	 *
	 * Since tanh'(delta*x) = delta * (1 - tanh(delta*x)^2)
	 * the value we return is extremely simple.
	 *
	 * hyperbolic_tan_derivative(x) is NOT tanh'(x).
	 */

    return n->delta * (1.0 - val * val);
}

AV* get_array(SV* aref)
{
    if (! is_array_ref(aref))
        croak("get_array() argument is not an array reference");
    
    return (AV*) SvRV(aref);
}

float get_float_element(AV* array, int index)
{
    SV **sva;
    SV *sv;

    sva = av_fetch(array, index, 0);
	if (!sva)
		return 0.0;

	sv = *sva;
	return SvNV(sv);
}

SV* get_element(AV* array, int index)
{
    SV   **temp;
    temp = av_fetch(array, index, 0);

    if (!temp)
        croak("Item %d in array is not defined", index);

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

NEURAL_NETWORK *c_get_network(int handle)
{
	NEURAL_NETWORK *n;

	if (handle < 0 || handle >= networks)
		croak("Invalid neural network handle");

	n = network[handle];

	if (n == NULL)
		croak("Stale neural network handle");

	return n;
}

int c_new_handle(void)
{
	int handle = -1;

	/*
	 * Allocate the network array if not already done.
	 * Then allocate a new handle for the network.
	 */

	if (network == NULL) {
		int i;

		networks = 10;
		network = malloc(networks * sizeof(*network));

		for (i = 0; i < networks; i++)
			network[i] = NULL;

		handle = 0;
	} else {
		int i;

		for (i = 0; i < networks; i++) {
			if (network[i] == NULL) {
				handle = i;
				break;
			}
		}

		if (handle == -1) {
			handle = networks;
			networks += 10;
			network = realloc(network, networks * sizeof(*network));

			for (i = networks - 10; i < networks; i++)
				network[i] = NULL;
		}
	}

	network[handle] = malloc(sizeof(NEURAL_NETWORK));

	return handle;
}

float c_get_learn_rate(int handle)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    return n->learn_rate;
}

void c_set_learn_rate(int handle, float rate)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    n->learn_rate = rate;
}

double c_get_delta(int handle)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    return n->delta;
}


void c_set_delta(int handle, double delta)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    n->delta = delta;
}

int c_get_use_bipolar(int handle)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    return n->use_bipolar;
}

void c_set_use_bipolar(int handle, int bipolar)
{
	NEURAL_NETWORK *n = c_get_network(handle);

    n->use_bipolar = bipolar;
}

int c_create_network(NEURAL_NETWORK *n)
{
    int i;
    /* each of the next two variables has an extra row for the "bias" */
    int input_layer_with_bias  = n->size.input  + 1;
    int hidden_layer_with_bias = n->size.hidden + 1;

    n->learn_rate = .2;
    n->delta = 1.0;
    n->use_bipolar = 0;

    n->tmp = malloc(sizeof(double) * n->size.input);

    n->neuron.input  = malloc(sizeof(double) * n->size.input);
    n->neuron.hidden = malloc(sizeof(double) * n->size.hidden);
    n->neuron.output = malloc(sizeof(double) * n->size.output);
    n->neuron.target = malloc(sizeof(double) * n->size.output);

    n->error.hidden  = malloc(sizeof(double) * n->size.hidden);
    n->error.output  = malloc(sizeof(double) * n->size.output);
    
    /* one extra for sentinel */
    n->weight.input_to_hidden  
        = malloc(sizeof(void *) * (input_layer_with_bias + 1));
    n->weight.hidden_to_output 
        = malloc(sizeof(void *) * (hidden_layer_with_bias + 1));

    if(!n->weight.input_to_hidden || !n->weight.hidden_to_output) {
        printf("Initial malloc() failed\n");
        return 0;
    }
    
    /* now allocate the actual rows */
    for(i = 0; i < input_layer_with_bias; i++) {
        n->weight.input_to_hidden[i] 
            = malloc(hidden_layer_with_bias * sizeof(double));
        if(n->weight.input_to_hidden[i] == 0) {
            free(*n->weight.input_to_hidden);
            printf("Second malloc() to weight.input_to_hidden failed\n");
            return 0;
        }
    }

    /* now allocate the actual rows */
    for(i = 0; i < hidden_layer_with_bias; i++) {
        n->weight.hidden_to_output[i] 
            = malloc(n->size.output * sizeof(double));
        if(n->weight.hidden_to_output[i] == 0) {
            free(*n->weight.hidden_to_output);
            printf("Second malloc() to weight.hidden_to_output failed\n");
            return 0;
        }
    }

    /* initialize the sentinel value */
    n->weight.input_to_hidden[input_layer_with_bias]   = 0;
    n->weight.hidden_to_output[hidden_layer_with_bias] = 0;

    return 1;
}

void c_destroy_network(int handle)
{
    double **row;
	NEURAL_NETWORK *n = c_get_network(handle);

    for(row = n->weight.input_to_hidden; *row != 0; row++) {
        free(*row);
    }
    free(n->weight.input_to_hidden);

    for(row = n->weight.hidden_to_output; *row != 0; row++) {
        free(*row);
    }
    free(n->weight.hidden_to_output);

    free(n->neuron.input);
    free(n->neuron.hidden);
    free(n->neuron.output);
    free(n->neuron.target);

    free(n->error.hidden);
    free(n->error.output);

	free(n->tmp);

	network[handle] = NULL;
}

/*
 * Build a Perl reference on array `av'.
 * This performs something like "$rv = \@av;" in Perl.
 */
SV *build_rv(AV *av)
{
	SV *rv;

	/*
	 * To understand what is going on here, look at retrieve_ref()
	 * in the Storable.xs file.  In particular, we don't perform
	 * an SvREFCNT_inc(av) because the av we're supplying is going
	 * to be referenced only by the REF we're building here.
	 *		--RAM
	 */

	rv = NEWSV(10002, 0);
	sv_upgrade(rv, SVt_RV);
	SvRV(rv) = (SV *) av;
	SvROK_on(rv);

	return rv;
}

/*
 * Build reference to a 2-dimensional array, implemented as an array
 * or array references.  The holding array has `rows' rows and each array
 * reference has `columns' entries.
 *
 * The name "axa" denotes the "product" of 2 arrays.
 */
SV *build_axaref(void *arena, int rows, int columns)
{
	AV *av;
	int i;
	double **p;

	av = newAV();
	av_extend(av, rows);

	for (i = 0, p = arena; i < rows; i++, p++) {
		int j;
		double *q;
		AV *av2;

		av2 = newAV();
		av_extend(av2, columns);

		for (j = 0, q = *p; j < columns; j++, q++)
			av_store(av2, j, newSVnv((NV) *q));

		av_store(av, i, build_rv(av2));
	}

	return build_rv(av);
}

#define EXPORT_VERSION	1
#define EXPORTED_ITEMS	9

/*
 * Exports the C data structures to the Perl world for serialization
 * by Storable.  We don't want to duplicate the logic of Storable here
 * even though we have to do some low-level Perl object construction.
 *
 * The structure we return is an array reference, which contains the
 * following items:
 *
 *  0	the export version number, in case format changes later
 * 	1	the amount of neurons in the input layer
 * 	2	the amount of neurons in the hidden layer
 * 	3	the amount of neurons in the output layer
 *	4	the learning rate
 *	5	the sigmoid delta
	6	whether to use a bipolar (tanh) routine instead of the sigmoid
 *	7	[[weight.input_to_hidden[0]], [weight.input_to_hidden[1]], ...]
 *	8	[[weight.hidden_to_output[0]], [weight.hidden_to_output[1]], ...]
 */
SV *c_export_network(int handle)
{
	NEURAL_NETWORK *n = c_get_network(handle);
	AV *av;
	SV *rv;
	int i = 0;

	av = newAV();
	av_extend(av, EXPORTED_ITEMS);

	av_store(av, i++,  newSViv(EXPORT_VERSION));
	av_store(av, i++,  newSViv(n->size.input));
	av_store(av, i++,  newSViv(n->size.hidden));
	av_store(av, i++,  newSViv(n->size.output));
	av_store(av, i++,  newSVnv(n->learn_rate));
	av_store(av, i++,  newSVnv(n->delta));
	av_store(av, i++,  newSViv(n->use_bipolar));
	av_store(av, i++,
				build_axaref(n->weight.input_to_hidden,
					n->size.input + 1, n->size.hidden + 1));
	av_store(av, i++,
				build_axaref(n->weight.hidden_to_output,
					n->size.hidden + 1, n->size.output));

	if (i != EXPORTED_ITEMS)
		croak("BUG in c_export_network()");

	return build_rv(av);
}

/*
 * Load a Perl array of array (a matrix) with "rows" rows and "columns" columns
 * into the pre-allocated C array of arrays.
 *
 * The "hold" argument is an holding array and the Perl array of array which
 * we expect is at index "idx" within that holding array.
 */
void c_load_axa(AV *hold, int idx, void *arena, int rows, int columns)
{
	SV **sav;
	SV *rv;
	AV *av;
	int i;
	double **array = arena;

	sav = av_fetch(hold, idx, 0);
	if (sav == NULL)
		croak("serialized item %d is not defined", idx);

	rv = *sav;
	if (!is_array_ref(rv))
		croak("serialized item %d is not an array reference", idx);

	av = get_array(rv);		/* This is an array of array refs */

	for (i = 0; i < rows; i++) {
		double *row = array[i];
		int j;
		AV *subav;

		sav = av_fetch(av, i, 0);
		if (sav == NULL)
			croak("serialized item %d has undefined row %d", idx, i);
		rv = *sav;
		if (!is_array_ref(rv))
			croak("row %d of serialized item %d is not an array ref", i, idx);

		subav = get_array(rv);

		for (j = 0; j < columns; j++)
			row[j] = get_float_element(subav, j);
	}
}

/*
 * Create new network from a retrieved data structure, such as the one
 * produced by c_export_network().
 */
int c_import_network(SV *rv)
{
	NEURAL_NETWORK *n;
	int handle;
	SV **sav;
	AV *av;
	int i = 0;

	/*
	 * Unfortunately, since those data come from the outside, we need
	 * to validate most of the structural information to make sure
	 * we're not fed garbage or something we cannot process, like a
	 * newer version of the serialized data. This makes the code heavy.
	 *		--RAM
	 */

	if (!is_array_ref(rv))
		croak("c_import_network() not given an array reference");

	av = get_array(rv);

	/* Check version number */
	sav = av_fetch(av, i++, 0);
	if (sav == NULL || SvIVx(*sav) != EXPORT_VERSION)
		croak("c_import_network() given unknown version %d",
			sav == NULL ? 0 : SvIVx(*sav));

	/* Check length -- at version 1, length is fixed to 13 */
	if (av_len(av) + 1 != EXPORTED_ITEMS)
		croak("c_import_network() not given a %d-item array reference",
			EXPORTED_ITEMS);

	handle = c_new_handle();
	n = c_get_network(handle);

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined input size (item %d)", i - 1);
    n->size.input  = SvIVx(*sav);

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined hidden size (item %d), i - 1");
    n->size.hidden = SvIVx(*sav);

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined output size (item %d)", i - 1);
    n->size.output = SvIVx(*sav);

    if (!c_create_network(n))
        return -1;

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined learn_rate (item %d)", i - 1);
    n->learn_rate = SvNVx(*sav);

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined delta (item %d)", i - 1);
    n->delta = SvNVx(*sav);

	sav = av_fetch(av, i++, 0);
	if (sav == NULL)
		croak("undefined use_bipolar (item %d)", i - 1);
    n->use_bipolar = SvIVx(*sav);

	c_load_axa(av, i++, n->weight.input_to_hidden,
		n->size.input + 1, n->size.hidden + 1);
	c_load_axa(av, i++, n->weight.hidden_to_output,
		n->size.hidden + 1, n->size.output);

    return handle;
}

/*
 * Support functions for back propogation
 */

void c_assign_random_weights(NEURAL_NETWORK *n)
{
    int hid, inp, out;

    for (inp = 0; inp < n->size.input + 1; inp++) {
        for (hid = 0; hid < n->size.hidden; hid++) {
            n->weight.input_to_hidden[inp][hid] = RAND_WEIGHT;
        }
    }

    for (hid = 0; hid < n->size.hidden + 1; hid++) {
        for (out = 0; out < n->size.output; out++) {
            n->weight.hidden_to_output[hid][out] = RAND_WEIGHT;
        }
    }
}

/*
 * Feed-forward Algorithm
 */

void c_feed_forward(NEURAL_NETWORK *n)
{
    int inp, hid, out;
    double sum;
	double (*activation)(NEURAL_NETWORK *, double);

	activation = n->use_bipolar ? hyperbolic_tan : sigmoid;

    /* calculate input to hidden layer */
    for (hid = 0; hid < n->size.hidden; hid++) {

        sum = 0.0;
        for (inp = 0; inp < n->size.input; inp++) {
            sum += n->neuron.input[inp]
                * n->weight.input_to_hidden[inp][hid];
        }

        /* add in bias */
        sum += n->weight.input_to_hidden[n->size.input][hid];

        n->neuron.hidden[hid] = (*activation)(n, sum);
    }

    /* calculate the hidden to output layer */
    for (out = 0; out < n->size.output; out++) {

        sum = 0.0;
        for (hid = 0; hid < n->size.hidden; hid++) {
            sum += n->neuron.hidden[hid] 
                * n->weight.hidden_to_output[hid][out];
        }

        /* add in bias */
        sum += n->weight.hidden_to_output[n->size.hidden][out];

        n->neuron.output[out] = (*activation)(n, sum);
    }
}

/*
 * Back-propogation algorithm.  This is where the learning gets done.
 */
void c_back_propagate(NEURAL_NETWORK *n)
{
    int inp, hid, out;
	double (*activation_derivative)(NEURAL_NETWORK *, double);

	activation_derivative = n->use_bipolar ?
		hyperbolic_tan_derivative : sigmoid_derivative;

    /* calculate the output layer error (step 3 for output cell) */
    for (out = 0; out < n->size.output; out++) {
        n->error.output[out] =
			(n->neuron.target[out] - n->neuron.output[out]) 
              * (*activation_derivative)(n, n->neuron.output[out]);
    }

    /* calculate the hidden layer error (step 3 for hidden cell) */
    for (hid = 0; hid < n->size.hidden; hid++) {

        n->error.hidden[hid] = 0.0;
        for (out = 0; out < n->size.output; out++) {
            n->error.hidden[hid] 
                += n->error.output[out] 
                 * n->weight.hidden_to_output[hid][out];
        }
        n->error.hidden[hid] 
            *= (*activation_derivative)(n, n->neuron.hidden[hid]);
    }

    /* update the weights for the output layer (step 4) */
    for (out = 0; out < n->size.output; out++) {
        for (hid = 0; hid < n->size.hidden; hid++) {
            n->weight.hidden_to_output[hid][out] 
                += (n->learn_rate 
                  * n->error.output[out] 
                  * n->neuron.hidden[hid]);
        }

        /* update the bias */
        n->weight.hidden_to_output[n->size.hidden][out] 
            += (n->learn_rate 
              * n->error.output[out]);
    }

    /* update the weights for the hidden layer (step 4) */
    for (hid = 0; hid < n->size.hidden; hid++) {

        for  (inp = 0; inp < n->size.input; inp++) {
            n->weight.input_to_hidden[inp][hid] 
                += (n->learn_rate 
                  * n->error.hidden[hid] 
                  * n->neuron.input[inp]);
        }

        /* update the bias */
        n->weight.input_to_hidden[n->size.input][hid] 
            += (n->learn_rate 
              * n->error.hidden[hid]);
    }
}

/*
 * Compute the Mean Square Error between the actual output and the
 * targeted output.
 */
double mean_square_error(NEURAL_NETWORK *n, double *target)
{
	double error = 0.0;
	int i;

	for (i = 0; i < n->size.output; i++)
		error += sqr(target[i] - n->neuron.output[i]);

	return 0.5 * error;
}

double c_train(int handle, SV* input, SV* output)
{
	NEURAL_NETWORK *n = c_get_network(handle);
    int i,length;
    AV *array;
    SV *elem;
    double *input_array  = malloc(sizeof(double) * n->size.input);
    double *output_array = malloc(sizeof(double) * n->size.output);
	double error;

    if (! is_array_ref(input) || ! is_array_ref(output)) {
        croak("train() takes two arrayrefs.");
    }
    
    array  = get_array(input);
    length = av_len(array)+ 1;
    
    if (length != n->size.input) {
        croak("Length of input array does not match network");
    }
    for (i = 0; i < length; i++) {
        input_array[i] = get_float_element(array, i);
    }

    array  = get_array(output);
    length = av_len(array) + 1;
    
    if (length != n->size.output) {
        croak("Length of output array does not match network");
    }
    for (i = 0; i < length; i++) {
        output_array[i] = get_float_element(array, i);
    }

    c_feed(n, input_array, output_array, 1);
	error = mean_square_error(n, output_array);

    free(input_array);
    free(output_array);

	return error;
}

int c_new_network(int input, int hidden, int output)
{
	NEURAL_NETWORK *n;
	int handle;

	handle = c_new_handle();
	n = c_get_network(handle);

    n->size.input  = input;
    n->size.hidden = hidden;
    n->size.output = output;

    if (!c_create_network(n))
        return -1;

    /* Perl already seeded the random number generator, via a rand(1) call */

    c_assign_random_weights(n);

    return handle;
}

double c_train_set(int handle, SV* set, int iterations, double mse)
{
	NEURAL_NETWORK *n = c_get_network(handle);
    AV     *input_array, *output_array; /* perl arrays */
    double *input, *output; /* C arrays */
    double max_error = 0.0;

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
        
        if (av_len(input_array)+1 != n->size.input)
            croak("Length of input data does not match");
        
        /* iterate over the input_array and assign the floats to input */
        
        for (j = 0; j < n->size.input; j++) {
            index = (i/2*n->size.input)+j;
            input[index] = get_float_element(input_array, j); 
        }
        
        output_array = get_array_from_aoa(set, i+1);
        if (av_len(output_array)+1 != n->size.output)
            croak("Length of output data does not match");

        for (j = 0; j < n->size.output; j++) {
            index = (i/2*n->size.output)+j;
            output[index] = get_float_element(output_array, j); 
        }
    }

    for (i = 0; i < iterations; i++) {
		max_error = 0.0;

        for (j = 0; j < (set_length/2); j++) {
			double error;

            c_feed(n, &input[j*n->size.input], &output[j*n->size.output], 1);

			if (mse >= 0.0 || i == iterations - 1) {
				error = mean_square_error(n, &output[j*n->size.output]);
				if (error > max_error)
					max_error = error;
			}
        }

		if (mse >= 0 && max_error <= mse)	/* Below their target! */
			break;
    }

    free(input);
    free(output);

	return max_error;
}

SV* c_infer(int handle, SV *array_ref)
{
	NEURAL_NETWORK *n = c_get_network(handle);
    int    i;
    AV     *perl_array, *result = newAV();

    /* feed the data */
    perl_array = get_array(array_ref);

    for (i = 0; i < n->size.input; i++)
        n->tmp[i] = get_float_element(perl_array, i);

    c_feed(n, n->tmp, NULL, 0); 

    /* read the results */
    for (i = 0; i < n->size.output; i++) {
        av_push(result, newSVnv(n->neuron.output[i]));
    }
    return newRV_noinc((SV*) result);
}

void c_feed(NEURAL_NETWORK *n, double *input, double *output, int learn)
{
    int i;

    for (i=0; i < n->size.input; i++) {
        n->neuron.input[i]  = input[i];
    }

    if (learn)
        for (i=0; i < n->size.output; i++)
            n->neuron.target[i] = output[i];

    c_feed_forward(n);

    if (learn) c_back_propagate(n); 
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
looking for.  We'll only have one neuron for the hidden layer.  Thus, we get a
network that resembles the following:

           Input   Hidden   Output

 input1  ----> n1 -+    +----> n4 --->  output1
                    \  /
                     n3
                    /  \
 input2  ----> n2 -+    +----> n5 --->  output2

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
 0   0   1    0

The type of network we use is a forward-feed back error propagation network,
referred to as a back-propagation network, for short.  The way it works is
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

By default, the activation function for the neurons is the sigmoid function
S() with delta = 1:

	S(x) = 1 / (1 + exp(-delta * x))

but you can change the delta after creation.  You can also use a bipolar
activation function T(), using the hyperbolic tangent:

	T(x) = tanh(delta * x)
	tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

which allows the network to have neurons negatively impacting the weight,
since T() is a signed function between (-1,+1) whereas S() only falls
within (0,1).

=head2 C<delta($delta)>

Fetches the current I<delta> used in activation functions to scale the
signal, or sets the new I<delta>. The higher the delta, the steeper the
activation function will be.  The argument must be strictly positive.

You should not change I<delta> during the traning.

=head2 C<use_bipolar($boolean)>

Returns whether the network currently uses a bipolar activation function.
If an argument is supplied, instruct the network to use a bipolar activation
function or not.

You should not change the activation function during the traning.

=head2 C<train(\@input, \@output)>

This method trains the network to associate the input data set with the output
data set.  Representing the "logical or" is as follows:

  $net->train([1,1] => [0,1]);
  $net->train([1,0] => [0,1]);
  $net->train([0,1] => [0,1]);
  $net->train([0,0] => [1,0]);

Note that a one pass through the data is seldom sufficient to train a network.
In the example "logical or" program, we actually run this data through the
network ten thousand times.

  for (1 .. 10000) {
    $net->train([1,1] => [0,1]);
    $net->train([1,0] => [0,1]);
    $net->train([0,1] => [0,1]);
    $net->train([0,0] => [1,0]);
  }

The routine returns the Mean Squared Error (MSE) representing how far the
network answered.

It is far preferable to use C<train_set()> as this lets you control the MSE
over the training set and it is more efficient because there are less memory
copies back and forth.

=head2 C<train_set(\@dataset, [$iterations, $mse])>

Similar to train, this method allows us to train an entire data set at once.
It is typically faster than calling individual "train" methods.  The first
argument is expected to be an array ref of pairs of input and output array
refs.

The second argument is the number of iterations to train the set.  If
this argument is not provided here, you may use the C<iterations()> method to
set it (prior to calling C<train_set()>, of course).  A default of 10,000 will
be provided if not set.

The third argument is the targeted Mean Square Error (MSE). When provided,
the traning sequence will compute the maximum MSE seen during an iteration
over the training set, and if it is less than the supplied target, the
training stops.  Computing the MSE at each iteration costs, but you are
certain to not over-train your network.

  $net->train_set([
    [1,1] => [0,1],
    [1,0] => [0,1],
    [0,1] => [0,1],
    [0,0] => [1,0],
  ], 10000, 0.01);

The routine returns the MSE of the last iteration, which is the highest MSE
seen over the whole training set (and not an average MSE).

=head2 C<iterations([$integer])>

If called with a positive integer argument, this method will allow you to set
number of iterations that train_set will use and will return the network
object.  If called without an argument, it will return the number of iterations
it was set to.

  $net->iterations;         # returns 100000
  my @training_data = ( 
    [1,1] => [0,1],
    [1,0] => [0,1],
    [0,1] => [0,1],
    [0,0] => [1,0],
  );
  $net->iterations(100000) # let's have lots more iterations!
      ->train_set(\@training_data);
  
=head2 C<learn_rate($rate)>)

This method, if called without an argument, will return the current learning
rate.  .20 is the default learning rate.

If called with an argument, this argument must be greater than zero and less
than one.  This will set the learning rate and return the object.
  
  $net->learn_rate; #returns the learning rate
  $net->learn_rate(.1)
      ->iterations(100000)
      ->train_set(\@training_data);

If you choose a lower learning rate, you will train the network slower, but you
may get a better accuracy.  A higher learning rate will train the network
faster, but it can have a tendancy to "overshoot" the answer when learning and
not learn as accurately.

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

=head1 AUTHORS

Curtis "Ovid" Poe, C<ovid [at] cpan [dot] org>

Multiple network support, persistence, export of MSE (mean squared error),
training until MSE below a given threshold and customization of the
activation function added by Raphael Manfredi C<Raphael_Manfredi@pobox.com>.

=head1 COPYRIGHT AND LICENSE

Copyright (c) 2003-2005 by Curtis "Ovid" Poe

Copyright (c) 2006 by Raphael Manfredi

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself. 

=cut
