"?x
BHostIDLE"IDLE1     \?@A     \?@a??	]?}??i??	]?}???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?S@9     ?S@A     ?S@I     ?S@ac_???^??i?sĵ ???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      H@9      H@A      H@I      H@az?!<7??i?? ??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      @@9      @@A      @@I      @@a??k}϶??i?????????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      9@9      9@A      9@I      9@a*m?3?z?i??^'????Unknown
dHostDataset"Iterator::Model(1     @Y@9     @Y@A      7@I      7@a??JD?x?i?0??",???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      4@I      4@a???\??t?i????U???Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      4@9      4@A      4@I      4@a???\??t?i?KZɴ???Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(1      2@9      2@A      2@I      2@a\Dm??r?i#~4P????Unknown
|
HostSelect"(binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@a?*?Vo?i@????????Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      (@9      (@A      (@I      (@az?!<7i?i??{?????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      (@9      (@A      (@I      (@az?!<7i?i???O?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      (@9      (@A      (@I      (@az?!<7i?iP??????Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@a7tL]?f?i??@??&???Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a7tL]?f?i???A?=???Unknown?
vHostSub"%binary_crossentropy/logistic_loss/sub(1      &@9      &@A      &@I      &@a7tL]?f?i?iٞ?T???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a\Dm??b?i9?FH?g???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??k}϶`?i??Tx???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a??k}϶`?i?ZA?
????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a??k}϶`?i?ƾ??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1       @9       @A       @I       @a??k}϶`?ie2<?x????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a9?|??]?i???{????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a9?|??]?i	?Wq?????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @az?!<7Y?i????A????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_8/ResourceApplyGradientDescent(1      @9      @A      @I      @az?!<7Y?i?Г??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_9/ResourceApplyGradientDescent(1      @9      @A      @I      @az?!<7Y?i??1?S????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @az?!<7Y?ii????????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @az?!<7Y?iAn?e???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @az?!<7Y?i????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @az?!<7Y?i?$?2x???Unknown
{HostMatMul"'gradient_tape/sequential/dense_4/MatMul(1      @9      @A      @I      @az?!<7Y?i?5HN,???Unknown
^ HostGatherV2"GatherV2(1      @9      @A      @I      @a???\??T?i(???s6???Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???\??T?i?????@???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???\??T?i?_SXK???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???\??T?iE?U?U???Unknown
?$HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a???\??T?i?&??<`???Unknown
?%HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???\??T?i?^خj???Unknown
?&HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???\??T?ib?!u???Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??k}϶P?iG?ˁ|}???Unknown
?(HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??k}϶P?i,Y??ׅ???Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??k}϶P?iIQ3????Unknown
?*HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a??k}϶P?i????????Unknown
V+HostSum"Sum_2(1      @9      @A      @I      @a??k}϶P?i?z? ?????Unknown
?,HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??k}϶P?i?0??E????Unknown
v-HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??k}϶P?i??C??????Unknown
b.HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??k}϶P?i??X?????Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??k}϶P?ioR??W????Unknown
?0HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??k}϶P?iT?'?????Unknown
?1HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??k}϶P?i9?>?????Unknown
?2HostBiasAddGrad"4gradient_tape/sequential/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??k}϶P?it??i????Unknown
t3Host_FusedMatMul"sequential/dense_4/BiasAdd(1      @9      @A      @I      @a??k}϶P?i*?^?????Unknown
o4HostSigmoid"sequential/dense_4/Sigmoid(1      @9      @A      @I      @a??k}϶P?i??z? ????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @az?!<7I?iT?ITe????Unknown
\6HostGreater"Greater(1      @9      @A      @I      @az?!<7I?i????????Unknown
e7Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @az?!<7I?i,??o?????Unknown?
V8HostMean"Mean(1      @9      @A      @I      @az?!<7I?i???2???Unknown
u9HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i
??w	???Unknown
?:HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @az?!<7I?ipU????Unknown
r;HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @az?!<7I?i?$? ???Unknown
v<HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @az?!<7I?iH#?4E???Unknown
v=HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @az?!<7I?i?+?"???Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i 4?P?(???Unknown
~?HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @az?!<7I?i?<`?/???Unknown
?@HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @az?!<7I?i?D/lW5???Unknown
}AHostMatMul")gradient_tape/sequential/dense_4/MatMul_1(1      @9      @A      @I      @az?!<7I?idM???;???Unknown
?BHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i?U͇?A???Unknown
qCHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @az?!<7I?i<^?%H???Unknown
?DHostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i?fk?iN???Unknown
qEHost_FusedMatMul"sequential/dense_3/Relu(1      @9      @A      @I      @az?!<7I?io:1?T???Unknown
?FHostReadVariableOp")sequential/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i?w	??Z???Unknown
?GHostReadVariableOp"(sequential/dense_4/MatMul/ReadVariableOp(1      @9      @A      @I      @az?!<7I?i??L7a???Unknown
tHHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??k}϶@?i?ڷ ee???Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a??k}϶@?i?5???i???Unknown
VJHostCast"Cast(1       @9       @A       @I       @a??k}϶@?iŐvh?m???Unknown
XKHostCast"Cast_3(1       @9       @A       @I       @a??k}϶@?i??U?q???Unknown
sLHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??k}϶@?i?F5?v???Unknown
|MHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??k}϶@?i???Iz???Unknown
?NHostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a??k}϶@?i???7w~???Unknown
dOHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??k}϶@?i?W?뤂???Unknown
jPHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??k}϶@?iw???҆???Unknown
vQHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??k}϶@?ij?S ????Unknown
}RHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??k}϶@?i]hq.????Unknown
`SHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??k}϶@?iP?P?[????Unknown
wTHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??k}϶@?iC0o?????Unknown
xUHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??k}϶@?i6y#?????Unknown
?VHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a??k}϶@?i)????????Unknown
?WHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??k}϶@?i/Ί????Unknown
?XHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??k}϶@?i??>@????Unknown
?YHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??k}϶@?i???m????Unknown
?ZHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a??k}϶@?i??l??????Unknown
?[Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a??k}϶@?i??KZɴ???Unknown
?\HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??k}϶@?i??*?????Unknown
]HostReluGrad")gradient_tape/sequential/dense_3/ReluGrad(1       @9       @A       @I       @a??k}϶@?i?P
?$????Unknown
?^HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??k}϶@?i???uR????Unknown
?_HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a??k}϶@?i??)?????Unknown
?`HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??k}϶@?i?a?ݭ????Unknown
vaHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??k}϶0?i ???????Unknown
XbHostCast"Cast_4(1      ??9      ??A      ??I      ??a??k}϶0?i?????????Unknown
XcHostCast"Cast_5(1      ??9      ??A      ??I      ??a??k}϶0?ijwk?????Unknown
adHostIdentity"Identity(1      ??9      ??A      ??I      ??a??k}϶0?i?gE	????Unknown?
?eHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??k}϶0?i?V ????Unknown
TfHostMul"Mul(1      ??9      ??A      ??I      ??a??k}϶0?i}rF?6????Unknown
?gHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??k}϶0?i?6?M????Unknown
whHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??k}϶0?io?%?d????Unknown
yiHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??k}϶0?i?z?{????Unknown
?jHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a??k}϶0?ia(a?????Unknown
?kHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??k}϶0?i???:?????Unknown
?lHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??k}϶0?iS???????Unknown
?mHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??k}϶0?i?0???????Unknown
?nHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??k}϶0?iE????????Unknown
?oHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??k}϶0?i????????Unknown
?pHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??k}϶0?i79?|????Unknown
?qHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a??k}϶0?i???V2????Unknown
?rHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??k}϶0?i)??0I????Unknown
?sHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??k}϶0?i?Ar
`????Unknown
~tHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a??k}϶0?i?a?v????Unknown
}uHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a??k}϶0?i??Q??????Unknown
vHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a??k}϶0?iJA??????Unknown
wHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1      ??9      ??A      ??I      ??a??k}϶0?i??0r?????Unknown
?xHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??k}϶0?i?? L?????Unknown
?yHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??k}϶0?ixR&?????Unknown
?zHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??k}϶0?i?????????Unknown*?x
sHostDataset"Iterator::Model::ParallelMapV2(1     ?S@9     ?S@A     ?S@I     ?S@a?r?????i?r??????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      H@9      H@A      H@I      H@a߼?xV4??i?K~?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      @@9      @@A      @@I      @@a)QΠ?E??i.?~?#????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      9@9      9@A      9@I      9@ah/??????i|?j?Y???Unknown
dHostDataset"Iterator::Model(1     @Y@9     @Y@A      7@I      7@aVJ?3(r??iOfq'????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      4@I      4@at??:W??i?#??m???Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(1      4@9      4@A      4@I      4@at??:W??i?A??S???Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      2@9      2@A      2@I      2@aO贁N??i????????Unknown
|	HostSelect"(binary_crossentropy/logistic_loss/Select(1      .@9      .@A      .@I      .@al?l???is?@?t???Unknown
z
HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      (@9      (@A      (@I      (@a߼?xV4??iA??S????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      (@9      (@A      (@I      (@a߼?xV4??i2Tv?????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      (@9      (@A      (@I      (@a߼?xV4??i?????????Unknown
XHostEqual"Equal(1      &@9      &@A      &@I      &@a?׍?????iZ??|?????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a?׍?????iט??????Unknown?
vHostSub"%binary_crossentropy/logistic_loss/sub(1      &@9      &@A      &@I      &@a?׍?????i*;L]n???Unknown
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@aO贁N??i??d?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a)QΠ?E??i????M???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1       @9       @A       @I       @a)QΠ?E??i!N&?֮???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1       @9       @A       @I       @a)QΠ?E??if???????Unknown
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1       @9       @A       @I       @a)QΠ?E??i??,?q???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1      @9      @A      @I      @a???=??iǒ_,?????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a???=??i?d?j????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a߼?xV4??i??tľc???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_8/ResourceApplyGradientDescent(1      @9      @A      @I      @a߼?xV4??i?:W?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_9/ResourceApplyGradientDescent(1      @9      @A      @I      @a߼?xV4??i??9xa????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a߼?xV4??i??2>???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a߼?xV4??i?{?+????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      @9      @A      @I      @a߼?xV4??i?????????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a߼?xV4??i?Q?ߦ???Unknown
{HostMatMul"'gradient_tape/sequential/dense_4/MatMul(1      @9      @A      @I      @a߼?xV4??i{??9xa???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @at??:W~?iF?7?&????Unknown
? HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @at??:W~?i??$?????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @at??:W~?i??[?????Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @at??:W~?i???2T???Unknown
?#HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @at??:W~?ir???????Unknown
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @at??:W~?i=???????Unknown
?%HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @at??:W~?iףp=
???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a)QΠ?Ex?i?s??:???Unknown
?'HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a)QΠ?Ex?iL'?Tk???Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a)QΠ?Ex?i??h$?????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a)QΠ?Ex?i?I??k????Unknown
V*HostSum"Sum_2(1      @9      @A      @I      @a)QΠ?Ex?i2??F?????Unknown
?+HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a)QΠ?Ex?iԂ-؂-???Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a)QΠ?Ex?ivoi^???Unknown
b-HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a)QΠ?Ex?i????????Unknown
?.HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a)QΠ?Ex?i?X??%????Unknown
?/HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)QΠ?Ex?i\?3?????Unknown
?0HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)QΠ?Ex?i??u?< ???Unknown
?1HostBiasAddGrad"4gradient_tape/sequential/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a)QΠ?Ex?i?.???P???Unknown
t2Host_FusedMatMul"sequential/dense_4/BiasAdd(1      @9      @A      @I      @a)QΠ?Ex?iB???S????Unknown
o3HostSigmoid"sequential/dense_4/Sigmoid(1      @9      @A      @I      @a)QΠ?Ex?i?g:b߱???Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a߼?xV4r?i^?+H????Unknown
\5HostGreater"Greater(1      @9      @A      @I      @a߼?xV4r?i????????Unknown
e6Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a߼?xV4r?iRi???Unknown?
V7HostMean"Mean(1      @9      @A      @I      @a߼?xV4r?i?=??C???Unknown
u8HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?iFs???g???Unknown
?9HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a߼?xV4r?i???oS????Unknown
r:HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a߼?xV4r?i:???????Unknown
v;HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a߼?xV4r?i???$????Unknown
v<HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a߼?xV4r?i.I?v?????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?i?~?#????Unknown
~>HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a߼?xV4r?i"???^B???Unknown
??HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a߼?xV4r?i???}?f???Unknown
}@HostMatMul")gradient_tape/sequential/dense_4/MatMul_1(1      @9      @A      @I      @a߼?xV4r?iz*0????Unknown
?AHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?i?Tkט????Unknown
qBHost_FusedMatMul"sequential/dense_2/Relu(1      @9      @A      @I      @a߼?xV4r?i
?\?????Unknown
?CHostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?i??M1j????Unknown
qDHost_FusedMatMul"sequential/dense_3/Relu(1      @9      @A      @I      @a߼?xV4r?i??>?????Unknown
?EHostReadVariableOp")sequential/dense_4/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?ix*0?;A???Unknown
?FHostReadVariableOp"(sequential/dense_4/MatMul/ReadVariableOp(1      @9      @A      @I      @a߼?xV4r?i?_!8?e???Unknown
tGHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a)QΠ?Eh?iC.? ?}???Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a)QΠ?Eh?i??b?/????Unknown
VIHostCast"Cast(1       @9       @A       @I       @a)QΠ?Eh?i???u????Unknown
XJHostCast"Cast_3(1       @9       @A       @I       @a)QΠ?Eh?i6??Z?????Unknown
sKHostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a)QΠ?Eh?i?gE#????Unknown
|LHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a)QΠ?Eh?i?5??F????Unknown
?MHostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1       @9       @A       @I       @a)QΠ?Eh?i)??????Unknown
dNHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a)QΠ?Eh?iz?'}?'???Unknown
jOHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a)QΠ?Eh?iˠ?E@???Unknown
vPHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a)QΠ?Eh?ioi^X???Unknown
}QHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a)QΠ?Eh?im=
ףp???Unknown
`RHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a)QΠ?Eh?i????????Unknown
wSHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a)QΠ?Eh?i?Kh/????Unknown
xTHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a)QΠ?Eh?i`??0u????Unknown
?UHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a)QΠ?Eh?i?v???????Unknown
?VHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a)QΠ?Eh?iE.? ????Unknown
?WHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a)QΠ?Eh?iSϊF???Unknown
?XHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a)QΠ?Eh?i??oS????Unknown
?YHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a)QΠ?Eh?i???2???Unknown
?ZHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a)QΠ?Eh?iF~??K???Unknown
?[HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a)QΠ?Eh?i?LR?]c???Unknown
\HostReluGrad")gradient_tape/sequential/dense_3/ReluGrad(1       @9       @A       @I       @a)QΠ?Eh?i??u?{???Unknown
?]HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a)QΠ?Eh?i9??>?????Unknown
?^HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1       @9       @A       @I       @a)QΠ?Eh?i??4/????Unknown
?_HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a)QΠ?Eh?iۅ??t????Unknown
v`HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a)QΠ?EX?i?%??????Unknown
XaHostCast"Cast_4(1      ??9      ??A      ??I      ??a)QΠ?EX?i-Tv??????Unknown
XbHostCast"Cast_5(1      ??9      ??A      ??I      ??a)QΠ?EX?iV??|?????Unknown
acHostIdentity"Identity(1      ??9      ??A      ??I      ??a)QΠ?EX?i"a ????Unknown?
?dHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a)QΠ?EX?i??gE#???Unknown
TeHostMul"Mul(1      ??9      ??A      ??I      ??a)QΠ?EX?i???)F???Unknown
?fHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a)QΠ?EX?i?Wi???Unknown
wgHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a)QΠ?EX?i#?X??%???Unknown
yhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a)QΠ?EX?iL&?֮1???Unknown
?iHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a)QΠ?EX?iu????=???Unknown
?jHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a)QΠ?EX?i??I??I???Unknown
?kHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a)QΠ?EX?i?[??V???Unknown
?lHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a)QΠ?EX?i???g:b???Unknown
?mHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a)QΠ?EX?i*;L]n???Unknown
?nHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a)QΠ?EX?iB??0?z???Unknown
?oHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a)QΠ?EX?ik???????Unknown
?pHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a)QΠ?EX?i?_,?Œ???Unknown
?qHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a)QΠ?EX?i??|??????Unknown
?rHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a)QΠ?EX?i?-??????Unknown
~sHostRealDiv")gradient_tape/binary_crossentropy/truediv(1      ??9      ??A      ??I      ??a)QΠ?EX?i??.????Unknown
}tHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a)QΠ?EX?i8?m?Q????Unknown
uHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a)QΠ?EX?iac?nt????Unknown
vHostReluGrad")gradient_tape/sequential/dense_2/ReluGrad(1      ??9      ??A      ??I      ??a)QΠ?EX?i??S?????Unknown
?wHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a)QΠ?EX?i?1_7?????Unknown
?xHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a)QΠ?EX?iܘ??????Unknown
?yHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a)QΠ?EX?i     ???Unknown2CPU