#!/usr/bin/perl

die "Usage: $0 ctm_prons output_dir" if(scalar(@ARGV) != 2);

$ctm_prons = $ARGV[0];
$output_dir = $ARGV[1];

$cmd = "mkdir -p $output_dir";
system($cmd) if(!-d $output_dir);

%data = (); #key=segt, value=phone start end ; phone start end ...
%data_txt = (); #key=segt, value=phones
open(CTM, $ctm_prons) || die "unable to open $ctm_prons\n";
while($l = <CTM>){
	chomp($l);
	@tab = split(/\s+/, $l);
	$segt = $tab[0];
	$start = $tab[1];
	$phDur = $tab[2];
	@pDur = split(/\,/, $phDur);
	for($i=4,$j=0; $i<scalar(@tab); $i++, $j++){
		$ph = $tab[$i];
		$ph =~ s/_[B|I|E|S]$//;
		$ph =~ s/[0-9]$//;
		$end = $start + $pDur[$j];
		$data{$segt}="" if(!exists($data{$segt}));
		$data_txt{$segt}="" if(!exists($data_txt{$segt}));
		$data{$segt} .= " ".$ph." $start $end ;";
		$data_txt{$segt}.= " ".$ph;
		$start = $end;
	}
}
close(CTM);

open(OALI, "> $output_dir/text_ali");
open(OTXT, "> $output_dir/text_phn");
foreach(sort keys %data){
	$segt = $_;
	print OALI $segt.$data{$segt}."\n";
	print OTXT $segt.$data_txt{$segt}."\n";
}
close(OTXT);
close(OALI);
