use Test::More 'no_plan';
use Test::Exception;
use strict;

my $CLASS;
BEGIN {
    unshift @INC => 'blib/lib/', '../blib/lib/';
    $CLASS = 'AI::NeuralNet::Simple';
    use_ok($CLASS) || die; 
};

can_ok($CLASS, 'new');

throws_ok {$CLASS->new}
    qr/^\QYou must supply three positive integers to new()\E/,
    '... and calling it without arguments should die';

throws_ok {$CLASS->new(qw/foo bar 2/)}
    qr/^\QArguments to new() must be positive integers\E/,
    '... and supplying new() with bad arguments should also die';

my $net = $CLASS->new(2,1,2);
ok($net, 'Calling new with good arguments should succeed');
isa_ok($net, $CLASS => '...and the object it returns');

can_ok($net, 'train');

# teach the network logical 'or'

ok($net->train([1,1], [0,1]), 'Calling train() with valid data should succeed');
for (1 .. 10000) {
    $net->train([1,1],[0,1]);
    $net->train([1,0],[0,1]);
    $net->train([0,1],[0,1]);
    $net->train([0,0],[1,0]);
}

can_ok($net, 'winner');
is($net->winner([1,1]), 1, '... and it should return the index of the highest valued result');
is($net->winner([1,0]), 1, '... and it should return the index of the highest valued result');
is($net->winner([0,1]), 1, '... and it should return the index of the highest valued result');
is($net->winner([0,0]), 0, '... and it should return the index of the highest valued result');

