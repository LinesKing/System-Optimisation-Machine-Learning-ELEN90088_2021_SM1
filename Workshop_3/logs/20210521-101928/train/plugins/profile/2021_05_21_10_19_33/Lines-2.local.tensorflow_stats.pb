"?e
BHostIDLE"IDLE1     ??@A     ??@a?Z??J???i?Z??J????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@aj??p??i??Q,K???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      S@9      S@A      S@I      S@a??Coul|?i?0$W???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      K@9      K@A      K@I      K@a???2t?i?nGR????Unknown
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@a$??0g?i?%=a?????Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      5@9      5@A      5@I      5@a??J?fj_?iD˩?m????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      3@9      3@A      3@I      3@a??Coul\?imaϣ????Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a???|?Z?i????????Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      0@I      0@a??8P??W?ik?fS????Unknown
s
Host_FusedMatMul"sequential_1/dense_4/Relu(1      .@9      .@A      .@I      .@a\K5??pV?iD??J????Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1      ,@9      ,@A      ,@I      ,@aν1???T?i?\?i?????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@a??*|??Q?iAr?=?????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      (@9      (@A      (@I      (@a??*|??Q?i??#?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a!'ǯtP?ij?????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      &@9      &@A      &@I      &@a!'ǯtP?i????+???Unknown
dHostDataset"Iterator::Model(1     ?U@9     ?U@A      $@I      $@a&G$n?M?il?s?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a???|?J?iiP??a???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a???|?J?if??[???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1      "@9      "@A      "@I      "@a???|?J?icp???!???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      "@9      "@A      "@I      "@a???|?J?i` .?(???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      "@9      "@A      "@I      "@a???|?J?i]?\yO/???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??8P??G?i??0\K5???Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??8P??G?iɬ?G;???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a??8P??G?i???!CA???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??8P??G?i5ɬ?G???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??8P??G?ik׀?:M???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @a??8P??G?i??T?6S???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1       @9       @A       @I       @a??8P??G?i??(?2Y???Unknown
?HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??8P??G?i??._???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aν1???D?i|?v?jd???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aν1???D?i??\?i???Unknown
? HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??*|??A?i?%G$n???Unknown
s!HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a??*|??A?i=0.1?r???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??*|??A?i?:Mw???Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a??*|??A?i?El?{???Unknown
?$HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??*|??A?i8P??????Unknown
?%HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a??*|??A?i?Z?ٔ????Unknown
?&HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??*|??A?i?e??????Unknown
v'Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @a??*|??A?i3p譎????Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a&G$n?=?i??L????Unknown
?)HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a&G$n?=?i??q?	????Unknown
?*HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a&G$n?=?i?
6?Ƙ???Unknown
?+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??8P??7?i???ě???Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??8P??7?i
????Unknown
v-HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??8P??7?i* t??????Unknown
v.HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??8P??7?iE'޼?????Unknown
?/HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a??8P??7?i`.H??????Unknown
?0HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??8P??7?i{5???????Unknown
?1HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??8P??7?i?<??????Unknown
?2HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1      @9      @A      @I      @a??8P??7?i?C???????Unknown
q3HostSigmoid"sequential_1/dense_5/Sigmoid(1      @9      @A      @I      @a??8P??7?i?J?s?????Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a??*|??1?i ????????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??*|??1?itU^1????Unknown
V6HostCast"Cast(1      @9      @A      @I      @a??*|??1?i???o????Unknown
X7HostEqual"Equal(1      @9      @A      @I      @a??*|??1?i`.H?????Unknown
V8HostMean"Mean(1      @9      @A      @I      @a??*|??1?ip?=??????Unknown
d9HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a??*|??1?i?jM2+????Unknown
z:HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a??*|??1?i?\?i????Unknown
v;HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??*|??1?ilul?????Unknown
v<HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @a??*|??1?i??{??????Unknown
?=HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a??*|??1?i??%????Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a??*|??1?ih?{c????Unknown
b?HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??*|??1?i?????????Unknown
x@HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @a??*|??1?i?e?????Unknown
~AHostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??*|??1?id???????Unknown
?BHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a??*|??1?i??O]????Unknown
CHostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      @9      @A      @I      @a??*|??1?i??ě????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??8P??'?i????????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a??8P??'?i(?R??????Unknown
XFHostCast"Cast_3(1       @9       @A       @I       @a??8P??'?i???????Unknown
uGHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??8P??'?iD????????Unknown
jHHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a??8P??'?iұq?????Unknown
rIHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??8P??'?i`?&??????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??8P??'?i??ۑ????Unknown
wKHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??8P??'?i|????????Unknown
?LHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??8P??'?i
?E?????Unknown
?MHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??8P??'?i???{?????Unknown
?NHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a??8P??'?i&ǯt????Unknown
?OHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a??8P??'?i??dm?????Unknown
?PHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a??8P??'?iB?f????Unknown
?QHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a??8P??'?i???^?????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??8P??'?i^ՃW????Unknown
?SHostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1       @9       @A       @I       @a??8P??'?i??8P?????Unknown
?THostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1       @9       @A       @I       @a??8P??'?iz??H
????Unknown
?UHostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @a??8P??'?i??A?????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??8P???i?a??H????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a??8P???i??W:????Unknown
XXHostCast"Cast_5(1      ??9      ??A      ??I      ??a??8P???i]e???????Unknown
aYHostIdentity"Identity(1      ??9      ??A      ??I      ??a??8P???i$?3?????Unknown?
TZHostMul"Mul(1      ??9      ??A      ??I      ??a??8P???i?hg?F????Unknown
|[HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??8P???i???+????Unknown
}\HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a??8P???iyl??????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??8P???i@?v$?????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??8P???ipѠD????Unknown
?_HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a??8P???i??+????Unknown
?`HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??8P???i?s???????Unknown
?aHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??8P???i\???????Unknown
?bHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??8P???i#w;?B????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??8P???i???????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??8P???i?z???????Unknown
?eHostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??8P???ix?J?????Unknown
?fHostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??8P???i?~??@????Unknown
?gHostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??8P???i     ???Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     P?@9     P?@A     P?@I     P?@a\#`ݑ,??i\#`ݑ,???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      S@9      S@A      S@I      S@a??{І???i??gJz????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      K@9      K@A      K@I      K@a<?/y???i???a?*???Unknown
iHostWriteSummary"WriteSummary(1      ?@9      ?@A      ?@I      ?@auU??`֖?i?fm????Unknown?
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      5@9      5@A      5@I      5@aД??????i??k?0]???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      3@9      3@A      3@I      3@a??{І???i?ۭ+????Unknown
gHostStridedSlice"strided_slice(1      2@9      2@A      2@I      2@a?Z?LW???iR??u@7???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      4@9      4@A      0@I      0@aÉ?C????iy??V?????Unknown
s	Host_FusedMatMul"sequential_1/dense_4/Relu(1      .@9      .@A      .@I      .@a'!?????i??y?????Unknown
s
Host_FusedMatMul"sequential_1/dense_3/Relu(1      ,@9      ,@A      ,@I      ,@a??M;????i?R??u@???Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      (@9      (@A      (@I      (@aR??2:???i}???.????Unknown
?Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      (@9      (@A      (@I      (@aR??2:???i?o??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a?~??
5??i\*ۻ???Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      &@9      &@A      &@I      &@a?~??
5??i???O???Unknown
dHostDataset"Iterator::Model(1     ?U@9     ?U@A      $@I      $@a4,?T?w}?ih.?r????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?Z?LW?z?i?&!?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?Z?LW?z?i?+?ϔ????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(1      "@9      "@A      "@I      "@a?Z?LW?z?i??W~?)???Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      "@9      "@A      "@I      "@a?Z?LW?z?i@)?,?^???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1      "@9      "@A      "@I      "@a?Z?LW?z?i???۴????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aÉ?C??w?i
5??????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aÉ?C??w?i? ????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @aÉ?C??w?i2O?&!???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aÉ?C??w?iFܦ?LP???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @aÉ?C??w?iZi.?r???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1       @9       @A       @I       @aÉ?C??w?in??~?????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1       @9       @A       @I       @aÉ?C??w?i??=o?????Unknown
?HostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aÉ?C??w?i??_????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??M;??t?i?;?%6???Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??M;??t?ixG??f_???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aR??2:?q?iG?9Â???Unknown
s HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aR??2:?q?i?}?????Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @aR??2:?q?i?D?!|????Unknown
V"HostSum"Sum_2(1      @9      @A      @I      @aR??2:?q?i??H??????Unknown
?#HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aR??2:?q?i???
5???Unknown
?$HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @aR??2:?q?iRB?3???Unknown
?%HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aR??2:?q?i!?y??V???Unknown
v&Host_FusedMatMul"sequential_1/dense_5/BiasAdd(1      @9      @A      @I      @aR??2:?q?i???gJz???Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a4,?T?wm?iN4???Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a4,?T?wm?iH??9????Unknown
?)HostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a4,?T?wm?it?݊?????Unknown
?*HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aÉ?C??g?i??!?D????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aÉ?C??g?i?Ke{????Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aÉ?C??g?i?sj???Unknown
v-HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aÉ?C??g?i???k?0???Unknown
?.HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @aÉ?C??g?i&?0d?H???Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aÉ?C??g?i?et\#`???Unknown
?0HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aÉ?C??g?i:,?T?w???Unknown
?1HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1      @9      @A      @I      @aÉ?C??g?i???LI????Unknown
q2HostSigmoid"sequential_1/dense_5/Sigmoid(1      @9      @A      @I      @aÉ?C??g?iN??Eܦ???Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aR??2:?a?i5?r?????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aR??2:?a?ic??8????Unknown
V5HostCast"Cast(1      @9      @A      @I      @aR??2:?a?i8???????Unknown
X6HostEqual"Equal(1      @9      @A      @I      @aR??2:?a?i?.?????Unknown
V7HostMean"Mean(1      @9      @A      @I      @aR??2:?a?i??=hC????Unknown
d8HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @aR??2:?a?i??p?????Unknown
z9HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @aR??2:?a?i???ܟ"???Unknown
v:HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @aR??2:?a?i?`?N4???Unknown
v;HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @aR??2:?a?im5	Q?E???Unknown
?<HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aR??2:?a?iT
<??W???Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @aR??2:?a?i;?n?Xi???Unknown
b>HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @aR??2:?a?i"???{???Unknown
x?HostCast"&gradient_tape/binary_crossentropy/Cast(1      @9      @A      @I      @aR??2:?a?i	??9?????Unknown
~@HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aR??2:?a?i?]tc????Unknown
?AHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @aR??2:?a?i?2:?????Unknown
BHostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1      @9      @A      @I      @aR??2:?a?i?m??????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aÉ?C??W?i??d?????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @aÉ?C??W?iHΰ?R????Unknown
XEHostCast"Cast_3(1       @9       @A       @I       @aÉ?C??W?i???\????Unknown
uFHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @aÉ?C??W?iҔ???????Unknown
jGHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @aÉ?C??W?ixU?????Unknown
rHHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @aÉ?C??W?i\[8?x???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @aÉ?C??W?i?>ZMB???Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @aÉ?C??W?i?!|? ???Unknown
?KHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @aÉ?C??W?i+?E?+???Unknown
?LHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @aÉ?C??W?ip????7???Unknown
?MHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @aÉ?C??W?i???=hC???Unknown
?NHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @aÉ?C??W?i???1O???Unknown
?OHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @aÉ?C??W?i??%6?Z???Unknown
?PHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @aÉ?C??W?i?uG??f???Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @aÉ?C??W?i?Xi.?r???Unknown
?RHostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1       @9       @A       @I       @aÉ?C??W?i<??W~???Unknown
?SHostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1       @9       @A       @I       @aÉ?C??W?iS?&!????Unknown
?THostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @aÉ?C??W?i?Ϣ?????Unknown
vUHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aÉ?C??G?i:??`ϛ???Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??aÉ?C??G?i????????Unknown
XWHostCast"Cast_5(1      ??9      ??A      ??I      ??aÉ?C??G?i~?ݘ????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??aÉ?C??G?i ??}????Unknown?
TYHostMul"Mul(1      ??9      ??A      ??I      ??aÉ?C??G?iº#Yb????Unknown
|ZHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??aÉ?C??G?id?4G????Unknown
}[HostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??aÉ?C??G?i?E?+????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aÉ?C??G?i??V?????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aÉ?C??G?iJ?gQ?????Unknown
?^HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??aÉ?C??G?i?rx?????Unknown
?_HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??aÉ?C??G?i?d?;????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??aÉ?C??G?i0V???????Unknown
?aHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??aÉ?C??G?i?G?I?????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??aÉ?C??G?it9?m????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??aÉ?C??G?i+??Q????Unknown
?dHostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aÉ?C??G?i?ރ6????Unknown
?eHostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aÉ?C??G?iZ?A????Unknown
?fHostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aÉ?C??G?i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU