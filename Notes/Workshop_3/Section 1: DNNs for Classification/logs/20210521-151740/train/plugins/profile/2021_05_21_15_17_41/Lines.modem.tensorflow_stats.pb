"?e
BHostIDLE"IDLE1     ܫ@A     ܫ@a?^?[J??i?^?[J???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?R@9     ?R@A     ?R@I     ?R@a?b???i?fl%????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?N@9     ?N@A     ?N@I     ?N@aiv\???i??ݺ???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @@9      @@A      @@I      @@a?@Û?iKb?X???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a?(0|?i%/?OP????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      9@9      9@A      9@I      9@a???Lx?i1C??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      8@9      8@A      8@I      8@a?p??Sw?i=#?̑????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      2@9      2@A      2@I      2@a`ԝ?~q?iF???????Unknown
o	HostSigmoid"sequential/dense_2/Sigmoid(1      2@9      2@A      2@I      2@a`ԝ?~q?iOs1??6???Unknown
|
HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a"??"?p?iW????W???Unknown
dHostDataset"Iterator::Model(1     ?V@9     ?V@A      0@I      0@a?@Ûo?i_'t??v???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      .@I      .@aJ?(m?if3{~ݓ???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      *@9      *@A      *@I      *@aQ???Ei?il?	#????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      &@9      &@A      &@I      &@aY<Kbe?iq h?????Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a?Z?pc?ivz	?????Unknown?
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@a?Z?pc?i{#Ԫf????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a?@Û_?iõ??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?@Û_?i?c?F????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?@Û_?i?y????Unknown
VHostMean"Mean(1      @9      @A      @I      @a??JH7[?i?o?8?%???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??JH7[?i????E3???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a??JH7[?i?G???@???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a??JH7[?i??%}N???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?p??SW?i??w'Z???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?p??SW?i?#??e???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?p??SW?i?[J{q???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?p??SW?i???%}???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?p??SW?i??	ψ???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?Z?pS?i???Y?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Z?pS?i??v??????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?Z?pS?i??#??????Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Z?pS?i???K?????Unknown
?!HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?Z?pS?i??}?h????Unknown
q"Host_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a?Z?pS?i??*? ????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?@ÛO?i????????Unknown
e$Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?@ÛO?i??;?????Unknown?
u%HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?@ÛO?i?S??t????Unknown
j&HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a?@ÛO?i?#??;????Unknown
j'HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?@ÛO?i???/????Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?@ÛO?i?????????Unknown
v)HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?@ÛO?i???}?????Unknown
v*HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?@ÛO?i?c?$V???Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?@ÛO?i?3??	???Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?@ÛO?i??r????Unknown
?-HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?@ÛO?i?Ӄ????Unknown
?.HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?@ÛO?iƣt?p ???Unknown
a/HostCast"sequential/Cast(1      @9      @A      @I      @a?@ÛO?i?seg7(???Unknown
t0Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a?@ÛO?i?CV?/???Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?p??SG?i?ߊ?5???Unknown
X2HostEqual"Equal(1      @9      @A      @I      @a?p??SG?i?{??;???Unknown
\3HostGreater"Greater(1      @9      @A      @I      @a?p??SG?i??}A???Unknown
?4HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?p??SG?iγ(RG???Unknown
?5HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?p??SG?i?O] 'M???Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?p??SG?i?????R???Unknown
~7HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?p??SG?iч???X???Unknown
?8HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?p??SG?i?#???^???Unknown
?9Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?p??SG?iӿ/?zd???Unknown
?:Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a?p??SG?i?[d?Oj???Unknown
?;HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?p??SG?i????$p???Unknown
?<HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a?p??SG?i֓???u???Unknown
?=HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?p??SG?i?/??{???Unknown
?>HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a?p??SG?i??6磁???Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?@Û??i?3?:?????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?@Û??iڛ'?j????Unknown
VAHostCast"Cast(1       @9       @A       @I       @a?@Û??i???M????Unknown
XBHostCast"Cast_3(1       @9       @A       @I       @a?@Û??i?k51????Unknown
XCHostCast"Cast_5(1       @9       @A       @I       @a?@Û??i?Ӑ?????Unknown
sDHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?@Û??i?;	??????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?@Û??iߣ?/ۜ???Unknown
rFHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?@Û??i????????Unknown
vGHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?@Û??i?sr֡????Unknown
?HHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?@Û??i???)?????Unknown
vIHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?@Û??i?Cc}h????Unknown
?JHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?@Û??i????K????Unknown
}KHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?@Û??i?T$/????Unknown
`LHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?@Û??i?{?w????Unknown
bMHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?@Û??i??D??????Unknown
xNHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?@Û??i?K?ٿ???Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?@Û??i??5r?????Unknown
?PHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?@Û??i??ş????Unknown
?QHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?@Û??i??&?????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?@Û??i???lf????Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?@Û??i?S?I????Unknown
}THostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?@Û??i-????Unknown
UHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?@Û??i?#g????Unknown
?VHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?@Û??i?????????Unknown
?WHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?@Û??i????????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?@Û/?i?'???????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a?@Û/?i?[qa?????Unknown
aZHostIdentity"Identity(1      ??9      ??A      ??I      ??a?@Û/?i??-?????Unknown?
T[HostMul"Mul(1      ??9      ??A      ??I      ??a?@Û/?i??鴝????Unknown
|\HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?@Û/?i???^?????Unknown
u]HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?@Û/?i?+b?????Unknown
w^HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?@Û/?i?_?r????Unknown
w_HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?@Û/?i???[d????Unknown
y`HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?@Û/?i?ǖV????Unknown
?aHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?@Û/?i??R?G????Unknown
?bHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?@Û/?i?/Y9????Unknown
?cHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?@Û/?i?c?+????Unknown
?dHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?@Û/?i񗇬????Unknown
?eHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?@Û/?i??CV????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?@Û/?i?????????Unknown
?gHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?@Û/?i???? ???Unknown
[hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i???? ???Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i???? ???Unknown*?d
sHostDataset"Iterator::Model::ParallelMapV2(1     ?R@9     ?R@A     ?R@I     ?R@aE??0??iE??0???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?N@9     ?N@A     ?N@I     ?N@a?j?2???i ??%????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @@9      @@A      @@I      @@aYg?з>??ik)??w???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a?m?????i w»T???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      9@9      9@A      9@I      9@a? ?????i8#?$????Unknown
lHostIteratorGetNext"IteratorGetNext(1      8@9      8@A      8@I      8@a???	???i???)???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      2@9      2@A      2@I      2@aD??ʎf??i,_?n????Unknown
oHostSigmoid"sequential/dense_2/Sigmoid(1      2@9      2@A      2@I      2@aD??ʎf??ip?׵???Unknown
|	HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a??M?Қ?iM\??c???Unknown
d
HostDataset"Iterator::Model(1     ?V@9     ?V@A      0@I      0@aYg?з>??i??u?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      .@I      .@a???S̪??ii??8???Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      *@9      *@A      *@I      *@a?s?Y????i	?a??????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      &@9      &@A      &@I      &@a'u_[??iA|]??g???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a/??e???iF?pR?????Unknown?
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@a/??e???iK$??0d???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @aYg?з>??i????+????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aYg?з>??i?w	?&.???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @aYg?з>??i"!L?!????Unknown
VHostMean"Mean(1      @9      @A      @I      @an?????i?u?
}????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @an?????i?? ??C???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @an?????i`[4????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @an?????i?r???????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a???	???i r'?K@???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???	???i6q??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a???	???ilp?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???	???i?o}2?#???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a???	???i?n?Y<o???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a/??e??i?y%Y????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a/??e??i???u????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a/??e??i?l???,???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a/??e??i???k???Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a/??e??i???S̪???Unknown
q!Host_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a/??e??i?j)?????Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aYg?з>y?i??ʎf???Unknown
e#Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aYg?з>y?i?l??N???Unknown?
u$HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aYg?з>y?iQina????Unknown
j%HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @aYg?з>y?i ???޳???Unknown
j&HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aYg?з>y?i?PM\????Unknown
v'HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aYg?з>y?i?g??????Unknown
v(HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aYg?з>y?i???,WK???Unknown
v)HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aYg?з>y?i\4??}???Unknown
?*HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aYg?з>y?i+f?R????Unknown
?+HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aYg?з>y?i??v{?????Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aYg?з>y?i??L???Unknown
?-HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aYg?з>y?i?d?Z?G???Unknown
a.HostCast"sequential/Cast(1      @9      @A      @I      @aYg?з>y?ig?Z?Gz???Unknown
t/Host_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aYg?з>y?i6?9Ŭ???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a???	?r?i??M?????Unknown
X1HostEqual"Equal(1      @9      @A      @I      @a???	?r?ilna?????Unknown
\2HostGreater"Greater(1      @9      @A      @I      @a???	?r?i'u_???Unknown
?3HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???	?r?i???=D???Unknown
?4HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a???	?r?i=??j???Unknown
z5HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???	?r?i?R??????Unknown
~6HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a???	?r?is?׵???Unknown
?7HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a???	?r?i?׵????Unknown
?8Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a???	?r?i?
}?????Unknown
?9Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a???	?r?iD
6?q'???Unknown
?:HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???	?r?i?	?PM???Unknown
?;HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a???	?r?iz	?&.s???Unknown
?<HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a???	?r?i	a:????Unknown
?=HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a???	?r?i?N?????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aYg?з>i?i??)????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @aYg?з>i?i~]??g????Unknown
V@HostCast"Cast(1       @9       @A       @I       @aYg?з>i?i??u?
???Unknown
XAHostCast"Cast_3(1       @9       @A       @I       @aYg?з>i?iL?\-?#???Unknown
XBHostCast"Cast_5(1       @9       @A       @I       @aYg?з>i?i?\-?#=???Unknown
sCHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aYg?з>i?i??bV???Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @aYg?з>i?i???T?o???Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aYg?з>i?i?[??????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aYg?з>i?iOp?????Unknown
?GHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @aYg?з>i?i??@|]????Unknown
vHHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aYg?з>i?i[4?????Unknown
?IHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aYg?з>i?i????????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aYg?з>i?i믲????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aYg?з>i?iRZ?[X ???Unknown
bLHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aYg?з>i?i?T?9???Unknown
xMHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @aYg?з>i?i ?$??R???Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aYg?з>i?i?Y??l???Unknown
?OHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aYg?з>i?i??:S????Unknown
?PHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @aYg?з>i?iU????????Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aYg?з>i?i?Xg?з???Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @aYg?з>i?i#8b????Unknown
}SHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aYg?з>i?i??N????Unknown
THostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @aYg?з>i?i?W?ь???Unknown
?UHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aYg?з>i?iX??????Unknown
?VHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aYg?з>i?i??zA
6???Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aYg?з>Y?i?c??B???Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??aYg?з>Y?i'WK?HO???Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??aYg?з>Y?i[?3U?[???Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??aYg?з>Y?i???h???Unknown
|[HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aYg?з>Y?i?V'u???Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??aYg?з>Y?i???hƁ???Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aYg?з>Y?i+??e????Unknown
w^HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??aYg?з>Y?i_V? ????Unknown
y_HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aYg?з>Y?i???|?????Unknown
?`HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aYg?з>Y?i? ??C????Unknown
?aHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aYg?з>Y?i?Uv4?????Unknown
?bHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aYg?з>Y?i/?^??????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aYg?з>Y?ic G?!????Unknown
?dHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??aYg?з>Y?i?U/H?????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aYg?з>Y?i˪?`????Unknown
?fHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aYg?з>Y?i?????????Unknown
[gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU