"?n
BHostIDLE"IDLE1     ??@A     ??@a?b?M???i?b?M????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     `b@9     `b@A     `b@I     `b@a??nⰛ?i??/^4????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @Z@9     @Z@A     @Z@I     @Z@a?O?|Ǔ?i?QREp????Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      @@9      @@A      @@I      @@a?r` ?x?i?S??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a?gW???u?iv??n]????Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      7@9      7@A      7@I      7@aRREp?Tq?iLԗ???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      5@9      5@A      5@I      5@aS?~???o?i??t,?2???Unknown
lHostIteratorGetNext"IteratorGetNext(1      3@9      3@A      3@I      3@a?r` ?l?i9=?,NO???Unknown
`	HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a`kZ?ךf?i????e???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      3@9      3@A      .@I      .@a`kZ?ךf?i??܃|???Unknown
dHostDataset"Iterator::Model(1      d@9      d@A      ,@I      ,@a7dT?e?isFV꜑???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      ,@9      ,@A      ,@I      ,@a7dT?e?iך??????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      *@9      *@A      *@I      *@a]N?C?c?i4??;M????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@a?UH?yb?i?17?b????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a?NB`??`?i?s?d?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a*?x??#^?i!??I????Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a*?x??#^?ii?/????Unknown?
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@a*?x??#^?i?(X,
???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aـl@6 [?i?^x/????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@aـl@6 [?i1??JL%???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@aـl@6 [?iq˸e?2???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@aـl@6 [?i?ـl@???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@aـl@6 [?i?7???M???Unknown
VHostMean"Mean(1       @9       @A       @I       @a?r` ?X?i*h??
Z???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?r` ?X?ic??=f???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?r` ?X?i????'r???Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1       @9       @A       @I       @a?r` ?X?i????5~???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a7dT?U?i#?f???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a7dT?U?i9M??N????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a7dT?U?ikw?t۝???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a7dT?U?i??z?g????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a7dT?U?i??Z??????Unknown
j!HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?UH?yR?i????????Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?UH?yR?i%??	????Unknown
}#HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @a?UH?yR?iP8??????Unknown
?$HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?UH?yR?i{\[u????Unknown
}%HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a?UH?yR?i??2*????Unknown
}&HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @a?UH?yR?iѤ??4????Unknown
a'HostCast"sequential/Cast(1      @9      @A      @I      @a?UH?yR?i?ț??????Unknown
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a*?x??#N?i ?;??????Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a*?x??#N?iDܐQ???Unknown
v*HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a*?x??#N?ih#|?????Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a*?x??#N?i?Avc???Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?r` ?H?i?Y??j???Unknown
X-HostEqual"Equal(1      @9      @A      @I      @a?r` ?H?i?q?q???Unknown
\.HostGreater"Greater(1      @9      @A      @I      @a?r` ?H?i㉜?x"???Unknown
?/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a?r` ?H?i ??(???Unknown
e0Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?r` ?H?i??@?.???Unknown?
?1HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?r` ?H?i:?i?4???Unknown
?2HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?r` ?H?iW꜑?:???Unknown
j3HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a?r` ?H?it??@???Unknown
~4HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?r` ?H?i????F???Unknown
?5HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?r` ?H?i?2?L???Unknown
?6HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?r` ?H?i?J?3?R???Unknown
?7HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?r` ?H?i?b\?X???Unknown
?8HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?r` ?H?i{???^???Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?UH?yB?i???Ec???Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?UH?yB?i/?]A?g???Unknown
V;HostCast"Cast(1      @9      @A      @I      @a?UH?yB?iD???Pl???Unknown
u<HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a?UH?yB?iY???p???Unknown
?=HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?UH?yB?in?}\[u???Unknown
v>HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a?UH?yB?i??ݺ?y???Unknown
v?HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?UH?yB?i??=f~???Unknown
b@HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?UH?yB?i??w?????Unknown
~AHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?UH?yB?i???p????Unknown
?BHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?UH?yB?i?/^4?????Unknown
?CHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a?UH?yB?i?A??{????Unknown
?DHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?UH?yB?iT? ????Unknown
?EHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?UH?yB?if~O?????Unknown
qFHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a?UH?yB?i+xޭ????Unknown
tGHost_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @a?UH?yB?i@?>?????Unknown
?HHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?UH?yB?iU??j????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?r` ?8?ic???????Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?r` ?8?iq??????Unknown
XKHostCast"Cast_3(1       @9       @A       @I       @a?r` ?8?i?^'!????Unknown
XLHostCast"Cast_4(1       @9       @A       @I       @a?r` ?8?i?̞?$????Unknown
XMHostCast"Cast_5(1       @9       @A       @I       @a?r` ?8?i???O(????Unknown
sNHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a?r` ?8?i???+????Unknown
rOHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?r` ?8?i??^x/????Unknown
vPHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?r` ?8?i???3????Unknown
zQHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?r` ?8?i?ߠ6????Unknown
}RHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?r` ?8?i?5:????Unknown
`SHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?r` ?8?i? _?=????Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?r` ?8?i?,?]A????Unknown
xUHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?r` ?8?i9??D????Unknown
?VHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?r` ?8?iE?H????Unknown
?WHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?r` ?8?i'Q_L????Unknown
~XHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?r` ?8?i5]??O????Unknown
}YHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?r` ?8?iCi?BS????Unknown
ZHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?r` ?8?iQu?V????Unknown
?[HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a?r` ?8?i_?_kZ????Unknown
?\HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1       @9       @A       @I       @a?r` ?8?im???]????Unknown
a]HostIdentity"Identity(1      ??9      ??A      ??I      ??a?r` ?(?it????????Unknown?
T^HostMul"Mul(1      ??9      ??A      ??I      ??a?r` ?(?i{?ߓa????Unknown
|_HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?r` ?(?i???]?????Unknown
d`HostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?r` ?(?i??(e????Unknown
?aHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?r` ?(?i?????????Unknown
ubHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?r` ?(?i??_?h????Unknown
wcHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?r` ?(?i????????Unknown
ydHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?r` ?(?i???Pl????Unknown
?eHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?r` ?(?i?ÿ?????Unknown
?fHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?r` ?(?i????o????Unknown
?gHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?r` ?(?i?????????Unknown
?hHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?r` ?(?i??ys????Unknown
?iHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?r` ?(?i???C?????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?r` ?(?i??_w????Unknown
?kHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?r` ?(?i????????Unknown
?lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?r` ?(?i?ퟡz????Unknown
mHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1      ??9      ??A      ??I      ??a?r` ?(?i???k?????Unknown
?nHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?r` ?(?i???5~????Unknown
?oHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?r` ?(?i?????????Unknown
epHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(i?????????Unknown
[qHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(i?????????Unknown
]rHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i?????????Unknown*?m
sHostDataset"Iterator::Model::ParallelMapV2(1     `b@9     `b@A     `b@I     `b@a????????i?????????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @Z@9     @Z@A     @Z@I     @Z@a?k"?k"??i?\?\???Unknown
oHostSigmoid"sequential/dense_3/Sigmoid(1      @@9      @@A      @@I      @@a????-???i?T???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      =@9      =@A      =@I      =@a???!ӡ?i"???J???Unknown?
oHost_FusedMatMul"sequential/dense/Relu(1      7@9      7@A      7@I      7@a?!?F??i???????Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      5@9      5@A      5@I      5@a???Й?iݶm۶m???Unknown
lHostIteratorGetNext"IteratorGetNext(1      3@9      3@A      3@I      3@a??`F[??ih?uAk????Unknown
`HostGatherV2"
GatherV2_1(1      .@9      .@A      .@I      .@a?1??zp??i??S?r
???Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      3@9      3@A      .@I      .@a?1??zp??i?z1?z1???Unknown
d
HostDataset"Iterator::Model(1      d@9      d@A      ,@I      ,@a?5?5??i?k"?k"???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      ,@9      ,@A      ,@I      ,@a?5?5??i???????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      *@9      *@A      *@I      *@a??i*???iƄPz?+???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@aB?ɯĀ??ik???????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      &@9      &@A      &@I      &@a?y?^??i?#???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@abB(=????i?1??zp???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@abB(=????i?????????Unknown?
?HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      $@9      $@A      $@I      $@abB(=????i?sǷ*5???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?n׃? ??i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?n׃? ??iX/?S/????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      "@9      "@A      "@I      "@a?n׃? ??i????>???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a?n׃? ??i???3????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?n׃? ??i?H>?????Unknown
VHostMean"Mean(1       @9       @A       @I       @a????-???i?b>?b>???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a????-???ih}h?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a????-???i֗?c?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1       @9       @A       @I       @a????-???iD??i*???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?5?5??id?;@o???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?5?5??i?`F[????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5?5??i?7?{?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a?5?5??i?Л?=???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?5?5??i????????Unknown
j HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aB?ɯĀ}?i?ytE?????Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aB?ɯĀ}?i??Ο????Unknown
}"HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      @9      @A      @I      @aB?ɯĀ}?i]?3X?3???Unknown
?#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aB?ɯĀ}?i05???n???Unknown
}$HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @aB?ɯĀ}?i??j?????Unknown
}%HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      @9      @A      @I      @aB?ɯĀ}?i?\R??????Unknown
a&HostCast"sequential/Cast(1      @9      @A      @I      @aB?ɯĀ}?i???}????Unknown
?'HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @abB(=??x?i.A,p?P???Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @abB(=??x?i???b?????Unknown
v)HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @abB(=??x?i8? U+????Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @abB(=??x?i?2?GW????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????-?s?i??0?????Unknown
X,HostEqual"Equal(1      @9      @A      @I      @a????-?s?i+M??3???Unknown
\-HostGreater"Greater(1      @9      @A      @I      @a????-?s?ibZZZZZ???Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a????-?s?i?gﵰ????Unknown
e/Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a????-?s?i?t?????Unknown?
?0HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a????-?s?i?m]????Unknown
?1HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a????-?s?i>??ȳ????Unknown
j2HostCast"binary_crossentropy/Cast(1      @9      @A      @I      @a????-?s?iu?C$
???Unknown
~3HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a????-?s?i???`F???Unknown
?4HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????-?s?i??m۶m???Unknown
?5HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????-?s?i?7????Unknown
?6HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????-?s?iQї?c????Unknown
?7HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????-?s?i??,??????Unknown
t8HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aB?ɯĀm?iq?ܲ:???Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aB?ɯĀm?iZr?w????Unknown
V:HostCast"Cast(1      @9      @A      @I      @aB?ɯĀm?iC<<<<<???Unknown
u;HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aB?ɯĀm?i,? ?Y???Unknown
?<HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aB?ɯĀm?iЛ?=w???Unknown
v=HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aB?ɯĀm?i??K??????Unknown
v>HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aB?ɯĀm?i?c?N?????Unknown
b?HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aB?ɯĀm?i?-??????Unknown
~@HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aB?ɯĀm?i??Z?@????Unknown
?AHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aB?ɯĀm?i??
??
???Unknown
?BHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @aB?ɯĀm?i???aB(???Unknown
?CHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aB?ɯĀm?itUj&?E???Unknown
?DHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aB?ɯĀm?i]?Cc???Unknown
qEHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @aB?ɯĀm?iF?ɯĀ???Unknown
tFHost_FusedMatMul"sequential/dense_3/BiasAdd(1      @9      @A      @I      @aB?ɯĀm?i/?ytE????Unknown
?GHostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aB?ɯĀm?i})9ƻ???Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a????-?c?i??fq????Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a????-?c?iP???????Unknown
XJHostCast"Cast_3(1       @9       @A       @I       @a????-?c?i????????Unknown
XKHostCast"Cast_4(1       @9       @A       @I       @a????-?c?i??S?r
???Unknown
XLHostCast"Cast_5(1       @9       @A       @I       @a????-?c?i$???Unknown
sMHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a????-?c?i???K?1???Unknown
rNHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a????-?c?i\+?ytE???Unknown
vOHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a????-?c?i??}?Y???Unknown
zPHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a????-?c?i?8H??l???Unknown
}QHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a????-?c?i0?v????Unknown
`RHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a????-?c?i?E?0!????Unknown
wSHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????-?c?iḩ^̧???Unknown
xTHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a????-?c?iSr?w????Unknown
?UHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a????-?c?i??<?"????Unknown
?VHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a????-?c?i<`??????Unknown
~WHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a????-?c?i???y????Unknown
}XHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a????-?c?itm?C$
???Unknown
YHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a????-?c?i?fq????Unknown
?ZHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1       @9       @A       @I       @a????-?c?i?z1?z1???Unknown
?[HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1       @9       @A       @I       @a????-?c?iH??%E???Unknown
a\HostIdentity"Identity(1      ??9      ??A      ??I      ??a????-?S?i?D?c?N???Unknown?
T]HostMul"Mul(1      ??9      ??A      ??I      ??a????-?S?i?????X???Unknown
|^HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a????-?S?i2˫??b???Unknown
d_HostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a????-?S?i??(|l???Unknown
?`HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a????-?S?i?Qv?Qv???Unknown
uaHostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a????-?S?i?[V'????Unknown
wbHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????-?S?ij?@??????Unknown
ycHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????-?S?i?&?ғ???Unknown
?dHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a????-?S?i_?????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a????-?S?iT???}????Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a????-?S?i???HS????Unknown
?gHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a????-?S?i?(??(????Unknown
?hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a????-?S?i>l?v?????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a????-?S?i????????Unknown
?jHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a????-?S?i??j??????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a????-?S?i(6P;????Unknown
lHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1      ??9      ??A      ??I      ??a????-?S?ivy5?T????Unknown
?mHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????-?S?iļi*????Unknown
?nHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????-?S?i	     ???Unknown
eoHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(i	     ???Unknown
[pHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(i	     ???Unknown
]qHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(i	     ???Unknown2CPU