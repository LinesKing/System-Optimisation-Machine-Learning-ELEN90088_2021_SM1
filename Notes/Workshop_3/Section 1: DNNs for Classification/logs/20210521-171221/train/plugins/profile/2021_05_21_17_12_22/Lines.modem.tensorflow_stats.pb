"?n
BHostIDLE"IDLE1     H?@A     H?@a?p?yz??i?p?yz???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Z@9     ?Z@A     ?Z@I     ?Z@ai_S??E??i??Dd???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      T@9      T@A      T@I      T@a#??/????iљi[???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?H@9     ?H@A      G@I      G@aB
V*K+??i??5x???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      @@9      @@A      @@I      @@aPk??J???i??E_????Unknown?
dHostDataset"Iterator::Model(1     ?`@9     ?`@A      <@I      <@a?{Rv?~?i??2dY????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ;@9      ;@A      ;@I      ;@ae??݋}?ii??q6???Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      5@9      5@A      5@I      5@a?ܽ??v?i#%#gd???Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(1      3@9      3@A      3@I      3@an?%???t?i?p???????Unknown
j
HostMean"binary_crossentropy/Mean(1      2@9      2@A      2@I      2@a??Yޓ?s?i?#J?a????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@a??5o?r?i?>???????Unknown
?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@a?{Rv?n?i3?+?:????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      *@9      *@A      *@I      *@aaN?$?sl?i?KPV????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a????&h?iu??|?-???Unknown?
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@a#??/??e?i;?Z?C???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a??Yޓ?c?i? ??VW???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??Yޓ?c?imz??	k???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aPk??J?a?i?;K̋|???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @aPk??J?a?iC??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aPk??J?a?i??da?????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @aPk??J?a?i???????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?{Rv?^?iW?,?d????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a?{Rv?^?i??g??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      @9      @A      @I      @a?{Rv?^?i????????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @a?{Rv?^?i%ްZ????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a? "?oCZ?i!??h|????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a? "?oCZ?i1G? ????Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a? "?oCZ?iAؚؿ???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a? "?oCZ?iQi???"???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a? "?oCZ?ia?mH0???Unknown
aHostCast"sequential/Cast(1      @9      @A      @I      @a? "?oCZ?iq?W %=???Unknown
e Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a#??/??U?iT??nH???Unknown?
?!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a#??/??U?i7}??S???Unknown
~"HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a#??/??U?ivL?]???Unknown
v#HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a#??/??U?i?n???h???Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a#??/??U?i?gO)?s???Unknown
{%HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a#??/??U?i?`???~???Unknown
t&Host_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @a#??/??U?i?Y?????Unknown
t'HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aPk??J?Q?i\??+?????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @aPk??J?Q?iQA????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aPk??J?Q?i?{Rv????Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @aPk??J?Q?i~ܘ?ì???Unknown
j+HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @aPk??J?Q?i4=???????Unknown
z,HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aPk??J?Q?i??%?E????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aPk??J?Q?i??k????Unknown
y.HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aPk??J?Q?iV_?0?????Unknown
?/HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aPk??J?Q?i??U?????Unknown
?0HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aPk??J?Q?i? ?{J????Unknown
?1HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @aPk??J?Q?ix???????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a? "?oCJ?i Jz|?????Unknown
V3HostCast"Cast(1      @9      @A      @I      @a? "?oCJ?i?oX-????Unknown
?4HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a? "?oCJ?i?c4?????Unknown
V5HostMean"Mean(1      @9      @A      @I      @a? "?oCJ?i??XO???Unknown
?6HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a? "?oCJ?i lM??
???Unknown
?7HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a? "?oCJ?i?4B?p???Unknown
?8HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a? "?oCJ?i0?6????Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a? "?oCJ?i??+?????Unknown
v:HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a? "?oCJ?i@? \#%???Unknown
`;HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a? "?oCJ?i?V8?+???Unknown
x<HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a? "?oCJ?iP
E2???Unknown
~=HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a? "?oCJ?i?????8???Unknown
?>HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a? "?oCJ?i`???f????Unknown
??HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a? "?oCJ?i?x???E???Unknown
?@Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a? "?oCJ?ipA݃?L???Unknown
~AHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a? "?oCJ?i?	?_S???Unknown
?BHostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a? "?oCJ?i???;?Y???Unknown
?CHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a? "?oCJ?i??;`???Unknown
?DHostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a? "?oCJ?i?c???f???Unknown
?EHostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a? "?oCJ?i,??\m???Unknown
?FHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a? "?oCJ?i?????s???Unknown
qGHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a? "?oCJ?i(???~z???Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aPk??J?A?i??1?~???Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @aPk??J?A?i?լ?????Unknown
XJHostCast"Cast_3(1       @9       @A       @I       @aPk??J?A?i9Nx??????Unknown
XKHostEqual"Equal(1       @9       @A       @I       @aPk??J?A?i?~? ????Unknown
TLHostMul"Mul(1       @9       @A       @I       @aPk??J?A?i﮾da????Unknown
sMHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @aPk??J?A?iJ?a??????Unknown
rNHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aPk??J?A?i??"????Unknown
vOHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @aPk??J?A?i @??????Unknown
vPHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @aPk??J?A?i[pK??????Unknown
?QHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @aPk??J?A?i???AD????Unknown
}RHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @aPk??J?A?iёԤ????Unknown
uSHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aPk??J?A?il5g????Unknown
bTHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aPk??J?A?i?1??e????Unknown
?UHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @aPk??J?A?i"b{?Ʒ???Unknown
?VHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @aPk??J?A?i}?'????Unknown
?WHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aPk??J?A?i?????????Unknown
?XHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @aPk??J?A?i3?dD?????Unknown
}YHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @aPk??J?A?i?#?H????Unknown
ZHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @aPk??J?A?i?S?i?????Unknown
?[HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aPk??J?A?iD?N?	????Unknown
?\HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aPk??J?A?i????j????Unknown
?]HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @aPk??J?A?i???!?????Unknown
v^HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aPk??J?1?i'}?j?????Unknown
X_HostCast"Cast_5(1      ??9      ??A      ??I      ??aPk??J?1?iT8?+????Unknown
a`HostIdentity"Identity(1      ??9      ??A      ??I      ??aPk??J?1?i????[????Unknown?
uaHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??aPk??J?1?i?E?F?????Unknown
|bHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aPk??J?1?i??,??????Unknown
dcHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??aPk??J?1?iv~??????Unknown
wdHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aPk??J?1?i5?"????Unknown
weHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??aPk??J?1?ib?!lM????Unknown
yfHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aPk??J?1?i?>s?}????Unknown
?gHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aPk??J?1?i?????????Unknown
?hHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??aPk??J?1?i?nH?????Unknown
?iHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??aPk??J?1?ih?????Unknown
?jHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??aPk??J?1?iC???>????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aPk??J?1?ip7$o????Unknown
?lHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??aPk??J?1?i??\m?????Unknown
mHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??aPk??J?1?i?g???????Unknown
?nHostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aPk??J?1?i?????????Unknown
+oHostCast"Cast_4(i?????????Unknown
[pHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown
[qHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i?????????Unknown
XrHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(i?????????Unknown*?m
sHostDataset"Iterator::Model::ParallelMapV2(1     ?Z@9     ?Z@A     ?Z@I     ?Z@a4y@???i4y@????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      T@9      T@A      T@I      T@av?g׺?i?????^???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?H@9     ?H@A      G@I      G@a????ޮ?i?8Uf=????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      @@9      @@A      @@I      @@a?W4y??i?c?La:???Unknown?
dHostDataset"Iterator::Model(1     ?`@9     ?`@A      <@I      <@a?,?M?ɢ?i:????????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ;@9      ;@A      ;@I      ;@a?!?S2??iv?g????Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      5@9      5@A      5@I      5@a??A??.??i??E?V????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      3@9      3@A      3@I      3@a??????i%?!?S2???Unknown
j	HostMean"binary_crossentropy/Mean(1      2@9      2@A      2@I      2@a???C(??iM??ش???Unknown
|
HostSelect"(binary_crossentropy/logistic_loss/Select(1      1@9      1@A      1@I      1@amA'?Ж?i	???????Unknown
?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@a?,?M?ɒ?itmA'???Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      *@9      *@A      *@I      *@a'AZir??i-ݷԲ???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@ai؁犆??i?"{??(???Unknown?
gHostStridedSlice"strided_slice(1      $@9      $@A      $@I      $@av?g׊?iE)L????Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a???C(??iO/???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a???C(??iY5S??U???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?W4y??i?:#s????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1       @9       @A       @I       @a?W4y??i@??W???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?W4y??isE?<W???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1       @9       @A       @I       @a?W4y??i?J?? ????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?,?M?ɂ?i?O?xH????Unknown
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      @9      @A      @I      @a?,?M?ɂ?i6T?epC???Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      @9      @A      @I      @a?,?M?ɂ?i?X5S?????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @a?,?M?ɂ?i?]k@?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a?g???i?a?+???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a?g???i?e???Z???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?g???i?i?Y????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a?g???i?m۶m????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?g???i?qw????Unknown
aHostCast"sequential/Cast(1      @9      @A      @I      @a?g???i?urD\???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @av?g?z?iy@?????Unknown?
? HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @av?g?z?iz|?????Unknown
~!HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @av?g?z?i??P????Unknown
v"HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @av?g?z?i0???2???Unknown
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @av?g?z?i??x?h???Unknown
{$HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @av?g?z?i??F]????Unknown
t%Host_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @av?g?z?iA?!????Unknown
t&HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?W4yu?i???R?????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a?W4yu?i?????)???Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?W4yu?iN?Y??T???Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a?W4yu?i???????Unknown
j*HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a?W4yu?i??)LǪ???Unknown
z+HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?W4yu?i[????????Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?W4yu?i
??ȫ ???Unknown
y-HostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?W4yu?i??a?+???Unknown
?.HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?W4yu?ih??E?V???Unknown
?/HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?W4yu?i?1??????Unknown
?0HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      @9      @A      @I      @a?W4yu?iƪ??t????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?g?p?iɬgq?????Unknown
V2HostCast"Cast(1      @9      @A      @I      @a?g?p?i̮5 ?????Unknown
?3HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a?g?p?iϰ????Unknown
V4HostMean"Mean(1      @9      @A      @I      @a?g?p?iҲ?}K-???Unknown
?5HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?g?p?iմ?,?M???Unknown
?6HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?g?p?iضm۶m???Unknown
?7HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?g?p?i۸;??????Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?g?p?i޺	9"????Unknown
v9HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?g?p?i????W????Unknown
`:HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?g?p?i侥??????Unknown
x;HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a?g?p?i??sE????Unknown
~<HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?g?p?i??A??.???Unknown
?=HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?g?p?i???.O???Unknown
?>HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?g?p?i???Qdo???Unknown
??Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?g?p?i?ȫ ?????Unknown
~@HostRealDiv")gradient_tape/binary_crossentropy/truediv(1      @9      @A      @I      @a?g?p?i??y?ϯ???Unknown
?AHostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?g?p?i??G^????Unknown
?BHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?g?p?i??;????Unknown
?CHostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?g?p?i????p???Unknown
?DHostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?g?p?iӱj?0???Unknown
?EHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a?g?p?i??P???Unknown
qFHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a?g?p?i?M?q???Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?W4ye?i`؁犆???Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?W4ye?i?ٵ????Unknown
XIHostCast"Cast_3(1       @9       @A       @I       @a?W4ye?i??%}????Unknown
XJHostEqual"Equal(1       @9       @A       @I       @a?W4ye?ih?E?????Unknown
TKHostMul"Mul(1       @9       @A       @I       @a?W4ye?i??Qdo????Unknown
sLHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?W4ye?i߅??????Unknown
rMHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?W4ye?ip๢a???Unknown
vNHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?W4ye?i????????Unknown
vOHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?W4ye?i ?!?S2???Unknown
?PHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?W4ye?ix?U ?G???Unknown
}QHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?W4ye?i???F]???Unknown
uRHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?W4ye?i(??>?r???Unknown
bSHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?W4ye?i???]8????Unknown
?THostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?W4ye?i??%}?????Unknown
?UHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?W4ye?i0?Y?*????Unknown
?VHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?W4ye?i?썻?????Unknown
?WHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?W4ye?i????????Unknown
}XHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?W4ye?i8????????Unknown
YHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1       @9       @A       @I       @a?W4ye?i??)	???Unknown
?ZHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?W4ye?i??]8????Unknown
?[HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?W4ye?i@??W4???Unknown
?\HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?W4ye?i???vzI???Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?W4yU?iD?_7T???Unknown
X^HostCast"Cast_5(1      ??9      ??A      ??I      ??a?W4yU?i?????^???Unknown
a_HostIdentity"Identity(1      ??9      ??A      ??I      ??a?W4yU?i???%?i???Unknown?
u`HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?W4yU?iH?-?lt???Unknown
|aHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?W4yU?i???D)???Unknown
dbHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?W4yU?i??a??????Unknown
wcHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?W4yU?iL??c?????Unknown
wdHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?W4yU?i????^????Unknown
yeHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?W4yU?i??/?????Unknown
?fHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?W4yU?iP??ش???Unknown
?gHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?W4yU?i??c??????Unknown
?hHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?W4yU?i???1Q????Unknown
?iHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?W4yU?iT???????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?W4yU?i ?1Q?????Unknown
?kHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?W4yU?i?????????Unknown
lHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a?W4yU?iX?epC????Unknown
?mHostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?W4yU?i     ???Unknown
+nHostCast"Cast_4(i     ???Unknown
[oHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i     ???Unknown
[pHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(i     ???Unknown
XqHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(i     ???Unknown2CPU