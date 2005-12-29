use 5.008;
use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    'NAME'		=> 'AI::NeuralNet::Simple',
    'VERSION_FROM'	=> 'Simple.pm', # finds $VERSION
    'PREREQ_PM'		=> {
        Inline            => '.44',
        'Log::Agent'      => '0.208',
        'Test::More'      => '0.48_01',
        'Test::Exception' => '0.15',
        'Sub::Uplevel'    => 0, # hate it, hate it, hate it
    },
    ($] >= 5.005 ?    ## Add these new keywords supported since 5.005
      (ABSTRACT_FROM => 'Simple.pm', # retrieve abstract from module
       AUTHOR     => 'Curtis "Ovid" Poe <poec@yahoo.com>') : ()),
);
